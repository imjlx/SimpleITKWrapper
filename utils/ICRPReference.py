#!/usr/bin/env python
# encoding: utf-8




def F18_bladder_cumulate_activity(age):
    if 0 <= age <= 3:
        activity = 0.16
    elif 3 < age <= 7:
        activity = 0.23
    else:
        activity = 0.26

    return activity


if __name__ == "__main__":
    print("Hi")
    pass
