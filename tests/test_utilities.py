import experimentsrpatch.utilities as ut

def test_get_logger():
    logger_1 = ut.get_logger()
    logger_2 = ut.get_logger()
    logger_3 = ut.get_logger(logfile_name='test_outputs/my_logger.log')
    logger_4 = ut.get_logger(logger_suffix='same')
    logger_5 = ut.get_logger(logger_suffix='same')
    
    assert logger_3.info('My log') == None
    assert logger_1 != logger_2
    assert logger_4 == logger_5
    
    