#!/usr/bin/python

import argparse

from pygments.lexers.asm import ObjdumpLexer


def main ():
    parser = argparse.ArgumentParser (description='Helloworld')
    MyModel.func_that_only_depends_on_class()

    if issubclass(object, MyModel):
        raise BaseException



###################################################
####                 DL model                   ###
###################################################

class MyModel:
    class_variable = None
    def __init__ (self):
        self.x = 0

    @classmethod
    def func_that_only_depends_on_class(cls):
        cls.class_variable = 0
        pass

    @staticmethod
    def staticDef(a, b):
        """This is a static method

        dsfk m;dkfm;dfm ;sdfm;dmf d;fmsd; fm;sdkfmd ;kfmsd;fm d;fkmdsfk msd;fm;dkfm ;sdfm
        df;dfdlf,' f'df'df dfldfls
        dfd'lfkd'lfmd;lmd;mf

        :arg: a: sdfddsfdf
        :b: sdffdsfds
        :return:
        """
        MyModel.class_variable = 9




###################################################
####                  MISC                      ###
###################################################

def isDead (*arg, **kwargs):
    return True


if __name__ == '__main__':

    main ()