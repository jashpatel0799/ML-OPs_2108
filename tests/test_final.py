import sys
# import os

sys.path.append('.')
from utils import train_dev_test_split
from plot_digit_classification import data_info

# sys.path.append('.')


def test_check_dataset_split_1b():
    data, label, train_frac, test_frac, dev_frac = data_info()
    rs = 25
    x_trn_1, y_trn_1, x_test_1, y_test_1, x_dev_1, y_dev_1 = train_dev_test_split(data, label, train_frac, test_frac, dev_frac, rs)
    x_trn_2, y_trn_2, x_test_2, y_test_2, x_dev_2, y_dev_2 = train_dev_test_split(data, label, train_frac, test_frac, dev_frac, rs)

    assert (x_trn_1 == x_trn_2).all()
    assert (y_trn_1 == y_trn_2).all()
    assert (x_test_1 == x_test_2).all()
    assert (y_test_1 == y_test_2).all()
    assert (x_dev_1 == x_dev_2).all()
    assert (y_dev_1 == y_dev_2).all()


def test_check_dataset_split_1c():
    data, label, train_frac, test_frac, dev_frac = data_info()
    rs1 = 25
    x_trn_1, y_trn_1, x_test_1, y_test_1, x_dev_1, y_dev_1 = train_dev_test_split(data, label, train_frac, test_frac, dev_frac, rs1)
    x_trn_2, y_trn_2, x_test_2, y_test_2, x_dev_2, y_dev_2 = train_dev_test_split(data, label, train_frac, test_frac, dev_frac)

    assert (x_trn_1 == x_trn_2).all() == False
    assert (y_trn_1 == y_trn_2).all() == False
    assert (x_test_1 == x_test_2).all() == False
    assert (y_test_1 == y_test_2).all() == False
    assert (x_dev_1 == x_dev_2).all() == False
    assert (y_dev_1 == y_dev_2).all() == False