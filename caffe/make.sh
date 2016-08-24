#!/bin/bash

make -j24 all && make -j24 test && build/test/test_all.testbin --gtest_list_tests
