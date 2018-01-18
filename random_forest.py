import numpy as np
from joblib import Parallel, delayed
import weave  # not in scipy any more - needs to be installed separately

class ForestParams:
    def __init__(self, num_classes, trees=50, depth=20, min_cnt=2, tests=5000):
        self.num_tests = tests
        self.min_sample_cnt = min_cnt
        self.max_depth = depth
        self.num_trees = trees
        self.bag_size = 0.8
        self.train_parallel = True
        self.num_classes = num_classes  # assumes that the classes are ordered from 0 to C


class Node:

    def __init__(self, node_id, node_cnt, exs_at_node, impurity, probability):
        self.node_id = node_id  # id of absolute node
        self.node_cnt = node_cnt  # id not including nodes that didn't get made
        self.exs_at_node = exs_at_node
        self.impurity = impurity
        self.num_exs = float(exs_at_node.shape[0])
        self.is_leaf = True
        self.info_gain = 0.0

        # output
        self.probability = probability.copy()
        self.class_id = probability.argmax()

        # node test
        self.test_ind1 = 0
        self.test_thresh = 0.0

    def update_node(self, test_ind1, test_thresh, info_gain):
        self.test_ind1 = test_ind1
        self.test_thresh = test_thresh
        self.info_gain = info_gain
        self.is_leaf = False

    def create_child(self, test_res, impurity, prob, child_type, node_cnt):
        # save absolute location in dataset
        inds_local = np.where(test_res)[0]
        inds = self.exs_at_node[inds_local]

        if child_type == 'left':
            self.left_node = Node(2*self.node_id+1, node_cnt, inds, impurity, prob)
        elif child_type == 'right':
            self.right_node = Node(2*self.node_id+2, node_cnt, inds, impurity, prob)

    def test(self, X):
        return X[self.test_ind1] < self.test_thresh

    def get_compact_node(self):
        # used for fast forest
        if not self.is_leaf:
            node_array = np.zeros(4)
            # dims 0 and 1 are reserved for indexing children
            node_array[2] = self.test_ind1
            node_array[3] = self.test_thresh
        else:
            node_array = np.zeros(2+self.probability.shape[0])
            node_array[0] = -1  # indicates that its a leaf
            node_array[1] = self.node_cnt  # the id of the node
            node_array[2:] = self.probability.copy()
        return node_array


class Tree:

    def __init__(self, tree_id, tree_params):
        self.tree_id = tree_id
        self.tree_params = tree_params
        self.num_nodes = 0
        self.compact_tree = None  # used for fast testing forest and small memory footprint

    def build_tree(self, X, Y, node):
        if (node.node_id < ((2.0**self.tree_params.max_depth)-1)) and (node.impurity > 0.0) \
                and (self.optimize_node(np.take(X, node.exs_at_node, 0), np.take(Y, node.exs_at_node), node)):
                self.num_nodes += 2
                self.build_tree(X, Y, node.left_node)
                self.build_tree(X, Y, node.right_node)

    def train(self, X, Y):

        # bagging
        exs_at_node = np.random.choice(Y.shape[0], int(Y.shape[0]*self.tree_params.bag_size), replace=False)
        exs_at_node.sort()

        # compute impurity
        prob, impurity = self.calc_impurity(np.take(Y, exs_at_node), np.ones((exs_at_node.shape[0], 1), dtype='bool'))

        # create root
        self.root = Node(0, 0, exs_at_node, impurity, prob[:, 0])
        self.num_nodes = 1

        # build tree
        self.build_tree(X, Y, self.root)

        # make compact version for fast testing
        self.compact_tree, _ = self.traverse_tree(self.root, np.zeros(0))

    def traverse_tree(self, node, compact_tree_in):
        node_loc = compact_tree_in.shape[0]
        compact_tree = np.hstack((compact_tree_in, node.get_compact_node()))

        # this assumes that the index for the left and right child nodes are the first two
        if not node.is_leaf:
            compact_tree, compact_tree[node_loc] = self.traverse_tree(node.left_node, compact_tree)
            compact_tree, compact_tree[node_loc+1] = self.traverse_tree(node.right_node, compact_tree)

        return compact_tree, node_loc

    def test(self, X):
        op = np.zeros((X.shape[0], self.tree_params.num_classes))

        # single dim test
        for ex_id in range(X.shape[0]):
            node = self.root
            while not node.is_leaf:
                if X[ex_id, node.test_ind1] < node.test_thresh:
                    node = node.right_node
                else:
                    node = node.left_node
            op[ex_id, :] = node.probability
        return op

    def test_fast(self, X):
        op = np.zeros((X.shape[0], self.tree_params.num_classes))
        tree = self.compact_tree  # work around

        #in memory: for non leaf  node - 0 is lchild index, 1 is rchild, 2 is dim to test, 3 is threshold
        #in memory: for leaf node - 0 is leaf indicator -1, 1 is the node id, the rest is the probability for each class
        code = """
        int ex_id, node_loc, c_it;
        for (ex_id=0; ex_id<NX[0]; ex_id++) {
            node_loc = 0;
            while (tree[node_loc] != -1) {
                if (X2(ex_id, int(tree[node_loc+2]))  <  tree[node_loc+3]) {
                    node_loc = tree[node_loc+1];  // right node
                }
                else {
                    node_loc = tree[node_loc];  // left node
                }

            }

            for (c_it=0; c_it<Nop[1]; c_it++) {
                OP2(ex_id, c_it) = tree[node_loc + 2 + c_it];
            }
        }
        """
        weave.inline(code, ['X', 'op', 'tree'])
        return op

    def get_leaf_ids(self, X):
        op = np.zeros((X.shape[0]))
        tree = self.compact_tree  # work around

        #in memory: for non leaf  node - 0 is lchild index, 1 is rchild, 2 is dim to test, 3 is threshold
        #in memory: for leaf node - 0 is leaf indicator -1, 1 is the node id, the rest is the probability for each class
        code = """
        int ex_id, node_loc;
        for (ex_id=0; ex_id<NX[0]; ex_id++) {
            node_loc = 0;
            while (tree[node_loc] != -1) {
                if (X2(ex_id, int(tree[node_loc+2]))  <  tree[node_loc+3]) {
                    node_loc = tree[node_loc+1];  // right node
                }
                else {
                    node_loc = tree[node_loc];  // left node
                }

            }

            op[ex_id] = tree[node_loc + 1];  // leaf id

        }
        """
        weave.inline(code, ['X', 'op', 'tree'])
        return op

    def calc_impurity(self, y_local, test_res):

        prob = np.zeros((self.tree_params.num_classes, test_res.shape[1]))

        # estimate probability
        for cc in range(self.tree_params.num_classes):
            node_test = test_res * (y_local[:, np.newaxis] == cc)
            prob[cc, :] = node_test.sum(axis=0)

        # normalize - make sure not to divide by zero
        prob[:, prob.sum(0) == 0] = 1.0
        prob = prob / prob.sum(0)

        # entropy
        #prob_log = prob.copy()
        #prob_log[np.where(prob_log == 0)] = np.nextafter(0, 1)
        #impurity = -np.sum(prob*np.log2(prob_log), axis=0)

        # gini
        impurity = 1.0-(prob**2).sum(0)

        return prob, impurity

    def node_split(self, x_local):
        # left node is false, right is true
        # single dim test
        test_inds_1 = np.sort(np.random.random_integers(0, x_local.shape[1]-1, self.tree_params.num_tests))
        x_local_expand = x_local.take(test_inds_1, 1)
        x_min = x_local_expand.min(0)
        x_max = x_local_expand.max(0)
        test_thresh = (x_max - x_min)*np.random.random_sample(self.tree_params.num_tests) + x_min
        #valid_var = (x_max != x_min)

        test_res = x_local_expand < test_thresh

        return test_res, test_inds_1, test_thresh

    def optimize_node(self, x_local, y_local, node):
        # perform split at node
        test_res, test_inds1, test_thresh = self.node_split(x_local)

        # count examples left and right
        num_exs_l = (~test_res).sum(axis=0).astype('float')
        num_exs_r = x_local.shape[0] - num_exs_l  # i.e. num_exs_r = test_res.sum(axis=0).astype('float')
        valid_inds = (num_exs_l >= self.tree_params.min_sample_cnt) & (num_exs_r >= self.tree_params.min_sample_cnt)

        successful_split = False
        if valid_inds.sum() > 0:
            # child node impurity
            prob_l, impurity_l = self.calc_impurity(y_local, ~test_res)
            prob_r, impurity_r = self.calc_impurity(y_local, test_res)

             # information gain - want the minimum
            num_exs_l_norm = num_exs_l/node.num_exs
            num_exs_r_norm = num_exs_r/node.num_exs
            #info_gain = - node.impurity + (num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l)
            info_gain = (num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l)

            # make sure we con only select from valid splits
            info_gain[~valid_inds] = info_gain.max() + 10e-10  # plus small constant
            best_split = info_gain.argmin()

            # create new child nodes and update current node
            node.update_node(test_inds1[best_split], test_thresh[best_split], info_gain[best_split])
            node.create_child(~test_res[:, best_split], impurity_l[best_split], prob_l[:, best_split], 'left', self.num_nodes+1)
            node.create_child(test_res[:, best_split], impurity_r[best_split], prob_r[:, best_split], 'right', self.num_nodes+2)

            successful_split = True

        return successful_split


## Parallel training helper - used to train trees in parallel
def train_forest_helper(t_id, X, Y, params, seed):
    #print 'tree', t_id
    np.random.seed(seed)
    tree = Tree(t_id, params)
    tree.train(X, Y)
    return tree


class Forest:

    def __init__(self, params):
        self.params = params
        self.trees = []

    def train(self, X, Y, delete_old_trees):
        if delete_old_trees:
            self.trees = []

        if self.params.train_parallel:
            # need to seed the random number generator for each process
            seeds = np.random.random_integers(0, 10e8, self.params.num_trees)
            self.trees.extend(Parallel(n_jobs=-1)(delayed(train_forest_helper)(t_id, X, Y, self.params, seeds[t_id])
                                             for t_id in range(self.params.num_trees)))
        else:
            #print 'Standard training'
            for t_id in range(self.params.num_trees):
                print 'tree', t_id
                tree = Tree(t_id, self.params)
                tree.train(X, Y)
                self.trees.append(tree)

    def test(self, X):
        op = np.zeros((X.shape[0], self.params.num_classes))
        for tt, tree in enumerate(self.trees):
            op_local = tree.test_fast(X)
            op += op_local
        op /= float(len(self.trees))
        return op

    def get_leaf_ids(self, X):
        op = np.zeros((X.shape[0], len(self.trees)), dtype=np.int64)
        for tt, tree in enumerate(self.trees):
            op[:, tt] = tree.get_leaf_ids(X)
        return op

    def delete_trees(self):
        del self.trees[:]
