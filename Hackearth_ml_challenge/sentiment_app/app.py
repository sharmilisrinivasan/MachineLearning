# -*- coding: utf-8 -*-
# @Author: sharmilis
# @Date:   2019-02-21 10:33:51
# @Last Modified by:   sharmilis
# @Last Modified time: 2019-02-21 21:39:35
from tornado.options import options
import tornado.web

import json
import pandas as pd

import conf_properties  # config parameters read from default config file
from constants import MiscConstants as Const
from controller import Controller

class Application(tornado.web.Application):
    """
    Main Tornado Application class to initialise tornado web application
    """

    def __init__(self):
        """
        Initialises tornado web application
        """
        tornado.web.Application.__init__(self,
                                         [ (options.end_point, Controller)],
                                         **get_app_settings())
        self.data_df = get_data_df()

def get_data_df():
    df_ = pd.DataFrame(json.load(open(options.review_data_file,Const.READ_FILE)))
    df_[Const.CONV_DATE_COL] = pd.to_datetime(df_[Const.ACTUAL_DATE_COL],
        format=Const.ACTUAL_DATE_FORMAT)
    return df_


def get_app_settings():
    """
    Returns: Dictionary of settings used by tornado application
    """
    return {}

if __name__ == "__main__":
    # Command-line parameters takes precedence over config from property file
    options.parse_command_line()

    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
