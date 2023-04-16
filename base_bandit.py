from collections import defaultdict
from copy import copy

import numpy as np
import pandas as pd

class BaseBandit:

    def get_history_values(self, attrs, hs):
        attrs_list = dict()
        for attr in attrs:
            attrs_list[attr] = np.array([getattr(h, attr) for h in hs])
        return attrs_list

    def get_history_cum_values(self, attrs, hs):
        summed_attrs_list = dict()
        for attr in attrs:
            l = [getattr(h, attr) for h in hs]
            summed_attrs_list[attr] = np.array(pd.Series(l).cumsum())
        return  summed_attrs_list

    def get_results(self, base_result, attrs, attrs_to_sum, intervals=100):
        attrs_list = self.get_history_values(attrs, self.history)
        summed_attrs_list = self.get_history_cum_values(attrs_to_sum, self.history)
        results = []
        for t in range(intervals, self.T+1, intervals):
            r = copy(base_result)
            r['time'] = t
            r['last_time_step'] = int(t + intervals > self.T)
            # import pdb; pdb.set_trace()
            for attr in attrs:
                r[attr] = attrs_list[attr][t]
            for attr in attrs_to_sum:
                r[attr + '_sum'] = summed_attrs_list[attr][t]
            results.append(r)
        return pd.DataFrame(results)

    def get_grouped_results(self, base_result, attrs,
        attrs_to_sum, grouped_attrs, intervals=100):
        attrs_list = self.get_history_values(attrs, self.history)
        summed_attrs_list = self.get_history_cum_values(attrs_to_sum, self.history)

        # grouped_histories = defaultdict(list)
        # for i, h in enumerate(self.history):
        #     grouped_histories[h.group].append(h)
        # grouped_attrs_list = dict()
        # for grp_name, hists in grouped_histories.items():
        #     grouped_attrs_list[grp_name] = self.get_history_cum_values(grouped_attrs, hists)

        results = []
        grouped_histories = defaultdict(list)
        for t, h in enumerate(self.history):
            grouped_histories[h.group].append(h)
            if t % intervals == 0 and t != 0:
                # print('adding!', t)
                r = copy(base_result)
                r['time'] = t
                r['last_time_step'] = int(t + intervals > self.T)
                for attr in attrs:
                    r[attr] = attrs_list[attr][t]
                for attr in attrs_to_sum:
                    r[attr + '_sum'] = summed_attrs_list[attr][t]

                for grp_name, hists in grouped_histories.items():
                    group_attrs_list = self.get_history_cum_values(grouped_attrs, grouped_histories[grp_name])
                    for attr in grouped_attrs:
                        r[attr + '_sum_g%s' % str(grp_name)] = group_attrs_list[attr][-1]
                        r['num_in_group_g%s' % str(grp_name)] = len(hists)
                results.append(r)
        return pd.DataFrame(results)