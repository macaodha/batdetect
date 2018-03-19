"""
Code taken from scipy.io.wavfile.py

Module to read wav files using numpy arrays

Functions
---------
`read`: Return the sample rate (in samples/sec) and data from a WAV file.
"""

from __future__ import division, print_function, absolute_import
import sys
import numpy
import struct
import warnings


__all__ = [
    'WavFileWarning',
    'read'
]


class WavFileWarning(UserWarning):
    pass


WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xfffe
KNOWN_WAVE_FORMATS = (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT)

# assumes file pointer is immediately
#  after the 'fmt ' id


def _read_fmt_chunk(fid, is_big_endian):
    """
    Returns
    -------
    size : int
        size of format subchunk in bytes (minus 8 for "fmt " and itself)
    format_tag : int
        PCM, float, or compressed format
    channels : int
        number of channels
    fs : int
        sampling frequency in samples per second
    bytes_per_second : int
        overall byte rate for the file
    block_align : int
        bytes per sample, including all channels
    bit_depth : int
        bits per sample
    """
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'

    size = res = struct.unpack(fmt+'I', fid.read(4))[0]
    bytes_read = 0

    if size < 16:
        raise ValueError("Binary structure of wave file is not compliant")

    res = struct.unpack(fmt+'HHIIHH', fid.read(16))
    bytes_read += 16

    format_tag, channels, fs, bytes_per_second, block_align, bit_depth = res

    if format_tag == WAVE_FORMAT_EXTENSIBLE and size >= (16+2):
        ext_chunk_size = struct.unpack(fmt+'H', fid.read(2))[0]
        bytes_read += 2
        if ext_chunk_size >= 22:
            extensible_chunk_data = fid.read(22)
            bytes_read += 22
            raw_guid = extensible_chunk_data[2+4:2+4+16]
            # GUID template {XXXXXXXX-0000-0010-8000-00AA00389B71} (RFC-2361)
            # MS GUID byte order: first three groups are native byte order,
            # rest is Big Endian
            if is_big_endian:
                tail = b'\x00\x00\x00\x10\x80\x00\x00\xAA\x00\x38\x9B\x71'
            else:
                tail = b'\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71'
            if raw_guid.endswith(tail):
                format_tag = struct.unpack(fmt+'I', raw_guid[:4])[0]
        else:
            raise ValueError("Binary structure of wave file is not compliant")

    if format_tag not in KNOWN_WAVE_FORMATS:
        raise ValueError("Unknown wave file format")

    # move file pointer to next chunk
    if size > (bytes_read):
        fid.read(size - bytes_read)

    return (size, format_tag, channels, fs, bytes_per_second, block_align,
            bit_depth)


# assumes file pointer is immediately after the 'data' id
def _read_data_chunk(fid, format_tag, channels, bit_depth, is_big_endian,
                     mmap=False):
    if is_big_endian:
        fmt = '>I'
    else:
        fmt = '<I'

    # Size of the data subchunk in bytes
    size = struct.unpack(fmt, fid.read(4))[0]

    # Number of bytes per sample
    bytes_per_sample = bit_depth//8
    if bit_depth == 8:
        dtype = 'u1'
    else:
        if is_big_endian:
            dtype = '>'
        else:
            dtype = '<'
        if format_tag == WAVE_FORMAT_PCM:
            dtype += 'i%d' % bytes_per_sample
        else:
            dtype += 'f%d' % bytes_per_sample
    if not mmap:
        data = numpy.fromstring(fid.read(size), dtype=dtype)
    else:
        start = fid.tell()
        data = numpy.memmap(fid, dtype=dtype, mode='c', offset=start,
                            shape=(size//bytes_per_sample,))
        fid.seek(start + size)

    if channels > 1:
        data = data.reshape(-1, channels)
    return data


def _skip_unknown_chunk(fid, is_big_endian):
    if is_big_endian:
        fmt = '>I'
    else:
        fmt = '<I'

    data = fid.read(4)
    # call unpack() and seek() only if we have really read data from file
    # otherwise empty read at the end of the file would trigger
    # unnecessary exception at unpack() call
    # in case data equals somehow to 0, there is no need for seek() anyway
    if data:
        size = struct.unpack(fmt, data)[0]
        fid.seek(size, 1)


def _read_riff_chunk(fid):
    str1 = fid.read(4)  # File signature
    if str1 == b'RIFF':
        is_big_endian = False
        fmt = '<I'
    elif str1 == b'RIFX':
        is_big_endian = True
        fmt = '>I'
    else:
        # There are also .wav files with "FFIR" or "XFIR" signatures?
        raise ValueError("File format {}... not "
                         "understood.".format(repr(str1)))

    # Size of entire file
    file_size = struct.unpack(fmt, fid.read(4))[0] + 8

    str2 = fid.read(4)
    if str2 != b'WAVE':
        raise ValueError("Not a WAV file.")

    return file_size, is_big_endian


def read(filename, mmap=False):
    """
    Open a WAV file

    Return the sample rate (in samples/sec) and data from a WAV file.

    Parameters
    ----------
    filename : string or open file handle
        Input wav file.
    mmap : bool, optional
        Whether to read data as memory-mapped.
        Only to be used on real files (Default: False).

        .. versionadded:: 0.12.0

    Returns
    -------
    rate : int
        Sample rate of wav file.
    data : numpy array
        Data read from wav file.  Data-type is determined from the file;
        see Notes.

    Notes
    -----
    This function cannot read wav files with 24-bit data.

    Common data types: [1]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    Note that 8-bit PCM is unsigned.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www-mmsp.ece.mcgill.ca/documents/audioformats/wave/Docs/riffmci.pdf

    """
    if hasattr(filename, 'read'):
        fid = filename
        mmap = False
    else:
        fid = open(filename, 'rb')

    try:
        file_size, is_big_endian = _read_riff_chunk(fid)
        fmt_chunk_received = False
        channels = 1
        bit_depth = 8
        format_tag = WAVE_FORMAT_PCM
        while fid.tell() < file_size:
            # read the next chunk
            chunk_id = fid.read(4)

            if not chunk_id:
                print("Unexpected end of file.")
                #raise ValueError("Unexpected end of file.")
                break
            elif len(chunk_id) < 4:
                raise ValueError("Incomplete wav chunk.")

            if chunk_id == b'fmt ':
                fmt_chunk_received = True
                fmt_chunk = _read_fmt_chunk(fid, is_big_endian)
                format_tag, channels, fs = fmt_chunk[1:4]
                bit_depth = fmt_chunk[6]
                if bit_depth not in (8, 16, 32, 64, 96, 128):
                    raise ValueError("Unsupported bit depth: the wav file "
                                     "has {}-bit data.".format(bit_depth))
            elif chunk_id == b'fact':
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id == b'data':
                if not fmt_chunk_received:
                    raise ValueError("No fmt chunk before data")
                data = _read_data_chunk(fid, format_tag, channels, bit_depth,
                                        is_big_endian, mmap)
            elif chunk_id == b'LIST':
                # Someday this could be handled properly but for now skip it
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id in (b'JUNK', b'Fake'):
                # Skip alignment chunks without warning
                _skip_unknown_chunk(fid, is_big_endian)
            else:
                warnings.warn("Chunk (non-data) not understood, skipping it.",
                              WavFileWarning)
                _skip_unknown_chunk(fid, is_big_endian)
    finally:
        if not hasattr(filename, 'read'):
            fid.close()
        else:
            fid.seek(0)

    return fs, data



if sys.version_info[0] >= 3:
    def _array_tofile(fid, data):
        # ravel gives a c-contiguous buffer
        fid.write(data.ravel().view('b').data)
else:
    def _array_tofile(fid, data):
        fid.write(data.tostring())

