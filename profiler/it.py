from io import BytesIO, TextIOWrapper, SEEK_SET
import os
import sys
import ctypes
import tempfile

f = BytesIO()

# Manually perform the redirection setup
original_stderr_fd = sys.stderr.fileno()
saved_stderr_fd = os.dup(original_stderr_fd)
tfile = tempfile.TemporaryFile(mode='w+b')
libc = ctypes.CDLL(None)

def _redirect_stderr(to_fd):
    libc.fflush(ctypes.c_void_p.in_dll(libc, 'stderr'))
    sys.stderr.close()
    os.dup2(to_fd, original_stderr_fd)
    sys.stderr = TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))

_redirect_stderr(tfile.fileno())

def get_import_times():
    # Manually perform the cleanup and revert stderr
    _redirect_stderr(saved_stderr_fd)
    tfile.flush()
    tfile.seek(0, SEEK_SET)
    f.write(tfile.read())
    tfile.close()
    os.close(saved_stderr_fd)

    return f.getvalue().decode('utf-8')

# Output the captured stderr
# print('START STDERR: \n{0}END STDERR'.format(f.getvalue().decode('utf-8')))
