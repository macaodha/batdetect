import pandas as pd
import numpy as np
import datetime as dt
import glob
import os


def save_to_txt(op_file, results):

    # takes a dictionary of results and saves to file
    with open(op_file, 'w') as file:
        head_str = 'file_name,detection_time,detection_prob'
        file.write(head_str + '\n')

        for ii in range(len(results)):
            for jj in range(len(results[ii]['prob'])):

                row_str = results[ii]['filename'] + ','
                tm = round(results[ii]['time'][jj],3)
                pr = round(results[ii]['prob'][jj],3)
                row_str += str(tm) + ',' + str(pr)
                file.write(row_str + '\n')


def create_audio_tagger_op(ip_file_name, op_file_name, st_times, det_confidence,
                           samp_rate, class_name):
    # saves the detections in an audiotagger friendly format

    col_names = ['Filename', 'Label', 'LabelTimeStamp', 'Spec_NStep',
                 'Spec_NWin', 'Spec_x1', 'Spec_y1', 'Spec_x2', 'Spec_y2',
                 'LabelStartTime_Seconds', 'LabelEndTime_Seconds',
                 'LabelArea_DataPoints', 'DetectorConfidence']

    nstep = 0.001
    nwin = 0.003
    call_width = 0.001  # code does not output call width so just make one up
    y_max = (samp_rate*nwin)/2.0
    num_calls = len(st_times)

    if num_calls == 0:
        da_at = pd.DataFrame(index=np.arange(0), columns=col_names)
    else:
        da_at = pd.DataFrame(index=np.arange(0, num_calls), columns=col_names)
        da_at['Spec_NStep'] = nstep
        da_at['Spec_NWin'] = nwin
        da_at['Label'] = 'bat'
        da_at['LabelTimeStamp'] = dt.datetime.now().isoformat()
        da_at['Spec_y1'] = 0
        da_at['Spec_y2'] = y_max
        da_at['Filename'] = ip_file_name

        for ii in np.arange(0, num_calls):

            st_time = st_times[ii]
            da_at.loc[ii, 'LabelStartTime_Seconds'] = np.round(st_time, 3)
            da_at.loc[ii, 'LabelEndTime_Seconds'] = np.round(st_time + call_width, 3)
            da_at.loc[ii, 'Label'] = class_name

            da_at.loc[ii, 'Spec_x1'] = np.round(st_time/nstep, 3)
            da_at.loc[ii, 'Spec_x2'] = np.round((st_time + call_width)/nstep, 3)

            da_at.loc[ii, 'DetectorConfidence'] = np.round(det_confidence[ii], 3)

    # save to disk
    da_at.to_csv(op_file_name, index=False)

    return da_at
