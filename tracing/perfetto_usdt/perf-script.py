# perf script event handlers, generated by perf script -g python
# Licensed under the terms of the GNU GPL License version 2

# The common_* event handler fields are the most useful fields common to
# all events.  They don't necessarily correspond to the 'common_*' fields
# in the format files.  Those fields not available as handler params can
# be retrieved using Python functions of the form common_*(context).
# See the perf-script-python Documentation for the list of available functions.

from __future__ import print_function

import os
import sys
import pandas as pd

sys.path.append(os.environ['PERF_EXEC_PATH'] + \
  '/scripts/python/Perf-Trace-Util/lib/Perf/Trace')

from perf_trace_context import *
from Core import *

time_start = 0
serialize_list = []
add_point_cloud_list = []
deserialize_list = []
get_point_cloud_list = []

def trace_begin():
  print("in trace_begin")

def trace_end():
  print("in trace_end")
  print("serialize")
  # for item in serialize_list:
  #   print(f'{item}ns')
  pd.set_option('float_format', '{:f}'.format)
  serialize_pd = pd.Series(serialize_list)
  print(f'min:{serialize_pd.min()}ns max:{serialize_pd.max()}ns mean:{serialize_pd.mean()}ns')
  print("add_point_cloud")
  add_point_cloud_pd = pd.Series(add_point_cloud_list)
  print(f'min:{add_point_cloud_pd.min()}ns max:{add_point_cloud_pd.max()}ns mean:{add_point_cloud_pd.mean()}ns')

  print("deserialize")
  # for item in deserialize_list:
  #   print(f'{item}ns')
  deserialize_pd = pd.Series(deserialize_list)
  print(f'min:{deserialize_pd.min()}ns max:{deserialize_pd.max()}ns mean:{deserialize_pd.mean()}ns')
  print("get_point_cloud")
  get_point_cloud_pd = pd.Series(get_point_cloud_list)
  print(f'min:{get_point_cloud_pd.min()}ns max:{get_point_cloud_pd.max()}ns mean:{get_point_cloud_pd.mean()}ns')

def sdt_normal__pb_add_point_cloud_begin(event_name, context, common_cpu,
  common_secs, common_nsecs, common_pid, common_comm,
  common_callchain, __probe_ip, perf_sample_dict):
  global time_start
  time_start = (common_secs * 1000000000) + common_nsecs

def sdt_normal__pb_add_point_cloud_end(event_name, context, common_cpu,
  common_secs, common_nsecs, common_pid, common_comm,
  common_callchain, __probe_ip, perf_sample_dict):
  global time_start
  global add_point_cloud_list
  end = (common_secs * 1000000000) + common_nsecs
  add_point_cloud_list += [end - time_start]

def sdt_normal__pb_get_point_cloud_begin(event_name, context, common_cpu,
  common_secs, common_nsecs, common_pid, common_comm,
  common_callchain, __probe_ip, perf_sample_dict):
  global time_start
  time_start = (common_secs * 1000000000) + common_nsecs

def sdt_normal__pb_get_point_cloud_end(event_name, context, common_cpu,
  common_secs, common_nsecs, common_pid, common_comm,
  common_callchain, __probe_ip, perf_sample_dict):
  global time_start
  global get_point_cloud_list
  end = (common_secs * 1000000000) + common_nsecs
  get_point_cloud_list += [end - time_start]

def sdt_normal__pb_serialize_start(event_name, context, common_cpu,
  common_secs, common_nsecs, common_pid, common_comm,
  common_callchain, __probe_ip, perf_sample_dict):
  global time_start
  time_start = (common_secs * 1000000000) + common_nsecs

def sdt_normal__pb_serialize_end(event_name, context, common_cpu,
  common_secs, common_nsecs, common_pid, common_comm,
  common_callchain, __probe_ip, perf_sample_dict):
  global time_start
  global serialize_list
  end = (common_secs * 1000000000) + common_nsecs
  serialize_list += [end - time_start]

def sdt_normal__pb_deserialize_start(event_name, context, common_cpu,
  common_secs, common_nsecs, common_pid, common_comm,
  common_callchain, __probe_ip, perf_sample_dict):
  global time_start
  time_start = (common_secs * 1000000000) + common_nsecs

def sdt_normal__pb_deserialize_fnish(event_name, context, common_cpu,
  common_secs, common_nsecs, common_pid, common_comm,
  common_callchain, __probe_ip, perf_sample_dict):
  global time_start
  global deserialize_list
  end = (common_secs * 1000000000) + common_nsecs
  deserialize_list += [end - time_start]

def trace_unhandled(event_name, context, event_fields_dict, perf_sample_dict):
    print(get_dict_as_string(event_fields_dict))
    print('Sample: {'+get_dict_as_string(perf_sample_dict['sample'], ', ')+'}')

def print_header(event_name, cpu, secs, nsecs, pid, comm):
  print("%-20s %5u %05u.%09u %8u %-20s " % \
  (event_name, cpu, secs, nsecs, pid, comm), end="")

def get_dict_as_string(a_dict, delimiter=' '):
  return delimiter.join(['%s=%s'%(k,str(v))for k,v in sorted(a_dict.items())])