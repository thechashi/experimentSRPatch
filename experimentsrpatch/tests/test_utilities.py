import utilities as ut

def test_get_logger():
    logger_1 = ut.get_logger()
    logger_2 = ut.get_logger()
    logger_3 = ut.get_logger(logfile_name='tests/test_outputs/my_logger.log')
    logger_4 = ut.get_logger(logger_suffix='same')
    logger_5 = ut.get_logger(logger_suffix='same')
    
    assert logger_3.info('My log') == None
    assert logger_1 != logger_2
    assert logger_4 == logger_5

def test_get_device_type():
    device = ut.get_device_type()
    assert device == 'cuda' or device == 'cpu'
    
def test_get_device_details():
    device, device_name = ut.get_device_details()
    assert type(device_name) == str
    
def test_random_image():
    r1 = ut.random_image(50)
    r2 = ut.random_image(50, False)
    r3 = ut.random_image(50, True, 2)
    assert r1.shape[0] == 1
    assert r1.shape[1] == 3
    assert r1.shape[2] == 50
    assert r1.shape[3] == 50
    assert r2.shape[0] == 3
    assert r3.shape[1] == 2

def test_get_image():
    assert True

def test_exception_handler():
    assert True


    
def test_test_image():
    assert True
    

    
def test_load_image():
    assert True
    
def test_load_grayscale_image():
    assert True
    
def test_npz_loader():
    assert True
    
def test_save_image():
    assert True

def test_get_gpu_details():
    assert True
    

    
def test_clear_cuda():
    assert True
    
def test_get_mean_std():
    assert True
    
def test_plot_data():
    assert True

def test_patch_count():
    assert True