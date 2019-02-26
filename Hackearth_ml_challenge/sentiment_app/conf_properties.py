# -*- coding: utf-8 -*-
# @Author: sharmilis
# @Date:   2019-02-21 21:28:51
# @Last Modified by:   sharmilis
# @Last Modified time: 2019-02-21 21:36:04
from tornado.options import define

define("end_point", default=r"/products")
define("port", default=8189)
define("response_type",default='application/json')
define("review_data_file", default="review_sentiment.json")
