from collections import defaultdict
from datetime import datetime
import functools

from tornado.options import options
from tornado.web import RequestHandler

from constants import FilterConstants as FilterConst
from constants import MiscConstants as MiscConst

class Controller(RequestHandler):

    def set_default_headers(self):
        self.set_header(MiscConst.HEADER_KEY_CONTENT_TYPE, options.response_type)

    def get(self):
        self.write(ControllerHelper(self.request.arguments, self.application.data_df).get_result_json())

class ControllerHelper(object):

    def __init__(self, arg_dict_, data_df_):
        self.arg_dict = arg_dict_
        self.filter_conditions = []
        self.count = FilterConst.COUNT_DEFUALT
        self.required_data = data_df_

    def _parse_args(self):
        to_return = defaultdict(lambda : [])
        for key,value in self.arg_dict.iteritems():
            to_return[key].extend(value)
        return to_return

    def _set_filter_conditions(self):
        filters_dict = self._parse_args()
        obtained_keys = filters_dict.keys()
        # 1. IN Filters
        for key in (set(obtained_keys) & set(FilterConst.POSSIBLE_IN_FILTER_KEYS)):
            self.filter_conditions.append(self.required_data[key].isin(filters_dict[key]))
        # 2. BOOL Filters
        for key in (set(obtained_keys) & set(FilterConst.POSSIBLE_BOOL_FILTER_KEYS)):
            check_val = ((filters_dict[key][0]).lower()==MiscConst.BOOL_TRUE)
            self.filter_conditions.append(self.required_data[key]==check_val)
        # 3. Date Filters
        if FilterConst.DATE_FILTER_KEY_FROM in obtained_keys:
            from_date = datetime.strptime(filters_dict[FilterConst.DATE_FILTER_KEY_FROM][0], MiscConst.REQ_DATE_FORMAT).date()
            self.filter_conditions.append(self.required_data[MiscConst.CONV_DATE_COL]>=from_date)
        if FilterConst.DATE_FILTER_KEY_TO in filters_dict:
            to_date = datetime.strptime(filters_dict[FilterConst.DATE_FILTER_KEY_TO][0], MiscConst.REQ_DATE_FORMAT).date()
            self.filter_conditions.append(self.required_data[MiscConst.CONV_DATE_COL]<=to_date)
        # 4. Count
        if FilterConst.COUNT_FILTER in obtained_keys:
            self.count = min(int(filters_dict[FilterConst.COUNT_FILTER][0]),FilterConst.COUNT_MAX)

    def _fetch_data(self):
        self._set_filter_conditions()
        if self.filter_conditions:
            filter_construct = functools.reduce(lambda c1,c2 : c1&c2,self.filter_conditions)
            self.required_data = self.required_data[filter_construct]
        self.required_data = self.required_data[:self.count]
        # Skipping custom column
        self.required_data = self.required_data.loc[:, self.required_data.columns != MiscConst.CONV_DATE_COL]

    def get_result_json(self):
        self._fetch_data()
        return self.required_data.to_json(orient=MiscConst.JSON_ORIENT)
