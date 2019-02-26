# -*- coding: utf-8 -*-
# @Author: sharmilis
# @Date:   2019-02-21 19:46:59
# @Last Modified by:   sharmilis
# @Last Modified time: 2019-02-21 21:16:22
class FilterConstants(object):
	COUNT_DEFUALT = 5
	COUNT_FILTER = "count"
	COUNT_MAX = 10
	DATE_FILTER_KEY_FROM="from"
	DATE_FILTER_KEY_TO="to"
	POSSIBLE_BOOL_FILTER_KEYS = ["verified_purchase"]
	POSSIBLE_IN_FILTER_KEYS=["colour", "size", "rating", "sentiment"]

class MiscConstants(object):
	ACTUAL_DATE_COL = "date"
	ACTUAL_DATE_FORMAT = "%d %B %Y"
	BOOL_TRUE = "true"
	CONV_DATE_COL = "conv_date"
	HEADER_KEY_CONTENT_TYPE = "Content-Type"
	JSON_ORIENT = "records"
	REQ_DATE_FORMAT = "%Y-%m-%d"
	READ_FILE = "r"
