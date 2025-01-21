#!/usr/bin/env python
# coding: utf-8

import unittest
import importlib
class MyTest(unittest.TestCase):
    def test_1(self):
        print("in MyTest.test_1")
    def test_0(self):
        print("in MyTest.test_0")

module = importlib.import_module("__main__")
members = dir(module)
tests = list()
for member in members:
    attr = getattr(module, member)
    if isinstance(attr, type) and issubclass(attr, unittest.TestCase):
        tests.append(attr)
# [MyTest]

print(tests)

for class_def in tests:
    obj = class_def() # MyTest()
    cases = list()
    attrs = dir(obj)
    for attr in attrs:
        print(attr)
        if attr.startswith("test") and callable(getattr(obj, attr)):
            cases.append(attr)
    print(cases)
    sorted_cases = sorted(cases) # ["test_0", "test_1"]
    for c in sorted_cases:
        func = getattr(obj, c)
        func()






