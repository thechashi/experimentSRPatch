import experimentsrpatch.binarysearch as bs
import experimentsrpatch.utilities as ut
import toml
import sys

if __name__ == "__main__":
    sys.excepthook = ut.exception_handler

    config = toml.load("batch_processing.toml")
    bs.do_binary_search(config["model"])
