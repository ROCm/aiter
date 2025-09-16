import torch


def get_num_sms():
    # Returns the Compute Unit count of the current device
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


def get_num_xcds():
    # Currently, you can't query this programmatically. For Mi300/Mi35x it's 8, so we hardcode that here.
    return 8
