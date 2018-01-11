
def is_gpu_available(cuda_only = True):

    from tensorflow.python.client import device_lib as _device_lib
    if cuda_only:
        return any((x.device_type == "GPU" )
                   for x in _device_lib.list_local_devices())
    else:
        return any((x.device_type == "GPU" or x.device_type == "SYCL")
                   for x in _device_lib.list_local_devices())



def main():
    x_l = is_gpu_available(True)
    print "Is there Cuda? ",x_l
    x_l = is_gpu_available(False)
    print "Others: ",x_l

    from tensorflow.python.client import device_lib as _device_lib
    print _device_lib.list_local_devices()

if __name__ == '__main__':
    main()
