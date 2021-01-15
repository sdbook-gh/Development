import yaml
import json
import re
import struct
from cloud import msg_def

TYPE = '_type'
TYPE_CLASS = '_class'
TYPE_FIELD = '_field'
PARENT = '_parent'
CHILD = '_child'
STRUCT = '_struct'
REPEAT = '_repeat'
REF = '_ref'
NODE = '_node'

regex_stringtype = re.compile('[0-9]+[s]')
regex_bytestype = re.compile('[0-9]+[B|b]')

def process_yaml(key: str, val: object, tree: dict, parent: object, brother: object) -> object:
    if isinstance(val, dict):
        node_type = TYPE_CLASS
    elif isinstance(val, str):
        node_type = TYPE_FIELD
    else:
        raise Exception(f'bad format of {key}:{str(key)}')
    tree[key] = {}
    tree[key][NODE] = key
    tree[key][TYPE] = node_type
    tree[key][PARENT] = parent
    if node_type == TYPE_CLASS:
        tree[key][CHILD] = {}
        if parent and brother[REPEAT] is None:
            raise Exception(f'{brother[NODE]} not set {REPEAT}')
        if brother and brother[REPEAT]:
            tree[key][REF] = brother
        item = None
        for classKey, classVal in val.items():
            item = process_yaml(classKey, classVal, tree[key][CHILD], tree[key], item)
    elif node_type == TYPE_FIELD:
        attr = json.loads(val)
        tree[key][STRUCT] = attr[STRUCT]
        if REPEAT in attr:
            tree[key][REPEAT] = attr[REPEAT]
    return tree[key]


def getClassName(node: dict) -> (str, bool):
    class_name: str = node[NODE]
    level = 0
    while True:
        if node[PARENT]:
            node = node[PARENT]
            class_name += '_'
            class_name += node[NODE]
            level += 1
        else:
            break
    return class_name, level > 0

def gen_msg_parse_code(val: dict):
    if CHILD not in val:
        raise Exception('bad yaml format, class field has no children')
    children = val[CHILD]

    #
    # class
    #
    class_name, class_is_sub = getClassName(val)
    print(f'''
class {class_name}:''')

    print(f'    __logger = logging.getLogger(\'{class_name}\')')

    #
    # __init__
    #
    if not class_is_sub:
        print(f'''
    def __init__(self):''')
    else:
        print(f'''
    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount''')
    print(f'        self.total_size = None')
    for field_name, field_info in children.items():
        if not class_is_sub:
            field_repeat = field_info[REPEAT] if REPEAT in field_info else None
            if field_repeat:
                print(f'        self.m_{field_name}:{field_repeat} = None')
            else:
                print(f'        self.m_{field_name} = None')
        else:
            print(f'        self.m_{field_name} = []')

    #
    # calc_real_size
    #
    print(f'''
    def calc_real_size(self) -> int:
        self.total_size = 0''')
    if not class_is_sub:
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_FIELD:
                field_struct = field_info[STRUCT]
                print(f'        self.total_size += struct.calcsize(\'{field_struct}\')')
            else:
                ref_field_name = field_info[REF][NODE]
                print(f'''
        if self.m_{ref_field_name} and self.m_{ref_field_name}.value > 0:
            if not self.m_{field_name}:
                raise Exception('m_{field_name} is not set')
            self.total_size += self.m_{field_name}.calc_real_size()''')
    else:
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_FIELD:
                field_struct = field_info[STRUCT]
                print(f'        self.total_size += struct.calcsize(\'{field_struct}\')')
        print(f'        self.total_size *= self.parentCount.value')
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_CLASS:
                ref_field_name = field_info[REF][NODE]
                print(f'''
        for i in range(len(self.m_{ref_field_name})):
            if self.m_{ref_field_name}[i] and self.m_{ref_field_name}[i].value > 0:
                if not self.m_{field_name}[i]:
                    raise Exception('m_{field_name}{{i}} is not set')
                self.total_size += self.m_{field_name}[i].calc_real_size()''')
    print(f'        return self.total_size')

    #
    # parse_buffer
    #
    print('''
    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos''')
    if not class_is_sub:
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_FIELD:
                field_struct = field_info[STRUCT]
                if regex_bytestype.match(field_struct):
                    print(f'        {field_name}_var = struct.unpack_from(\'{field_struct}\', buffer, pos)')
                else:
                    print(f'        {field_name}_var = struct.unpack_from(\'{field_struct}\', buffer, pos)[0]')
                print(f'        {class_name}.__logger.debug(f\'parse get {field_name}_var:{{{field_name}_var}}\')')
                print(f'        self.m_{field_name} = {field_name}_var')
                print(f'        pos += struct.calcsize(\'{field_struct}\')')
            else:
                child_class_name, child_class_is_sub = getClassName(field_info)
                ref_field_name = field_info[REF][NODE]
                print(f'''
        if self.m_{ref_field_name} and self.m_{ref_field_name}.value > 0:
            self.m_{field_name} = {child_class_name}(self.m_{ref_field_name})
            if pos + self.m_{field_name}.calc_real_size() > len(buffer):
                raise Exception('start_pos + size of m_{field_name} > buffer size')
            pos = self.m_{field_name}.parse_buffer(buffer, pos)''')
    else:
        print(f'        for i in range(self.parentCount.value):')
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_FIELD:
                field_struct = field_info[STRUCT]
                if regex_bytestype.match(field_struct):
                    print(f'            {field_name}_var = struct.unpack_from(\'{field_struct}\', buffer, pos)')
                else:
                    print(f'            {field_name}_var = struct.unpack_from(\'{field_struct}\', buffer, pos)[0]')
                print(f'            self.m_{field_name}.append({field_name}_var)')
                print(f'            {class_name}.__logger.debug(f\'parse get {field_name}_var:{{{field_name}_var}}\')')
                print(f'            pos += struct.calcsize(\'{field_struct}\')')
            else:
                child_class_name, child_class_is_sub = getClassName(field_info)
                ref_field_name = field_info[REF][NODE]
                print(f'''
            if self.m_{ref_field_name}[-1] and self.m_{ref_field_name}[-1].value > 0:
                self.m_{field_name}.append({child_class_name}(self.m_{ref_field_name}[-1]))
                if pos + self.m_{field_name}[-1].calc_real_size() > len(buffer):
                    raise Exception('start_pos + size of m_{field_name} > buffer size')
                pos = self.m_{field_name}[-1].parse_buffer(buffer, pos)''')
    print(f'        return pos')

    #
    # fill_buffer
    #
    print(f'''
    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos''')
    if not class_is_sub:
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_FIELD:
                field_struct = field_info[STRUCT]
                field_repeat = field_info[REPEAT] if REPEAT in field_info else None
                if field_repeat:
                    print(f'        {field_name}_var = self.m_{field_name}.value')
                else:
                    print(f'        {field_name}_var = self.m_{field_name}')
                if regex_bytestype.match(field_struct):
                    print(f'        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + \'{field_struct}\', buffer, pos, *{field_name}_var)')
                else:
                    print(f'        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + \'{field_struct}\', buffer, pos, {field_name}_var)')
                print(f'        {class_name}.__logger.debug(f\'fill {field_name}_var:{{{field_name}_var}}\')')
                print(f'        pos += struct.calcsize(\'{field_struct}\')')
            else:
                ref_field_name = field_info[REF][NODE]
                print(f'''
        if self.m_{ref_field_name} and self.m_{ref_field_name}.value > 0:
            if not self.m_{field_name}:
                raise Exception('m_{field_name} is not set')
            pos = self.m_{field_name}.fill_buffer(buffer, pos)''')
    else:
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_FIELD:
                print(f'''
        if len(self.m_{field_name}) != self.parentCount.value:
            raise Exception('parentCount mismatch m_{field_name} count')''')
        print(f'        for i in range(self.parentCount.value):')
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_FIELD:
                field_struct = field_info[STRUCT]
                field_repeat = field_info[REPEAT] if REPEAT in field_info else None
                if field_repeat:
                    print(f'            {field_name}_var = self.m_{field_name}[i].value')
                else:
                    print(f'            {field_name}_var = self.m_{field_name}[i]')
                if regex_bytestype.match(field_struct):
                    print(f'            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + \'{field_struct}\', buffer, pos, *{field_name}_var)')
                else:
                    print(f'            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + \'{field_struct}\', buffer, pos, {field_name}_var)')
                print(f'            {class_name}.__logger.debug(f\'fill {field_name}_var:{{{field_name}_var}}\')')
                print(f'            pos += struct.calcsize(\'{field_struct}\')')
            else:
                ref_field_name = field_info[REF][NODE]
                print(f'''
            if self.m_{ref_field_name}[i] and self.m_{ref_field_name}[i].value > 0:
                if not self.m_{field_name}[i]:
                    raise Exception('m_{field_name} is not set')
                pos = self.m_{field_name}[i].fill_buffer(buffer, pos)''')
    print(f'        return pos')

    for field_name, field_info in children.items():
        if field_info[TYPE] == TYPE_CLASS:
            gen_msg_parse_code(field_info)


char = '_'
byte = '0x01,'


def gen_msg_test_code(val: dict):
    class_name, class_is_sub = getClassName(val)
    children = val[CHILD]
    class_var = class_name.lower() + '_var'
    if not class_is_sub:
        print(f'    # {class_var} = cloud.{class_name}()')
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_FIELD:
                field_struct = field_info[STRUCT]
                field_repeat = field_info[REPEAT] if REPEAT in field_info else None
                if field_repeat:
                    print(f'    # {class_var}.m_{field_name} = ctypes.c_int(0)')
                    continue
                match_strtype = regex_stringtype.match(field_struct)
                match_barrtype = regex_bytestype.match(field_struct)
                if match_strtype:
                    count = int(match_strtype.group()[:-1])
                    print(f'    {class_var}.m_{field_name} = b\'' + (char * count) + '\'')
                elif match_barrtype:
                    count = int(match_barrtype.group()[:-1])
                    print(f'    {class_var}.m_{field_name} = (' + (byte * count) + ')')
                elif struct.calcsize(field_struct) == 1:
                    if field_struct == 's':
                        print(f'    {class_var}.m_{field_name} = b\'_\'')
                    else:
                        print(f'    {class_var}.m_{field_name} = 0x01')
                elif struct.calcsize(field_struct) == 2:
                    print(f'    {class_var}.m_{field_name} = 0x0102')
                elif struct.calcsize(field_struct) == 4:
                    print(f'    {class_var}.m_{field_name} = 0x01020304')
                elif struct.calcsize(field_struct) == 8:
                    print(f'    {class_var}.m_{field_name} = 0x0102030405060708')
            else:
                gen_msg_test_code(field_info)
        print(f'''
    total_size = {class_var}.calc_real_size()
    buffer = bytearray(total_size)
    pos = {class_var}.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)
''')
    else:
        parent_class = val[PARENT]
        parent_class_name, parent_class_is_sub = getClassName(parent_class)
        ref_field = val[REF]
        ref_field_name = ref_field[NODE]
        parent_class_var = f'{parent_class_name.lower()}_var'
        ref_field_count_var = f'{parent_class_var}.m_{ref_field_name}'
        if parent_class_is_sub:
            print(f'    # {class_var} = cloud.{class_name}({ref_field_count_var}[])')
            print(f'    # {parent_class_var}.m_{val[NODE]}.append({class_var})')
        else:
            print(f'    # {class_var} = cloud.{class_name}({ref_field_count_var})')
            print(f'    # {parent_class_var}.m_{val[NODE]} = {class_var}')
        first_field_name = None
        for field_name, field_info in children.items():
            if field_info[TYPE] == TYPE_FIELD:
                if not first_field_name:
                    first_field_name = field_name
                field_struct = field_info[STRUCT]
                field_repeat = field_info[REPEAT] if REPEAT in field_info else None
                if field_repeat:
                    print(f'    # {class_var}.m_{field_name}.append(ctypes.c_int(0))')
                    continue
                match_strtype = regex_stringtype.match(field_struct)
                if match_strtype:
                    count = int(match_strtype.group()[:-1])
                    print(f'    {class_var}.m_{field_name}.append(b\'' + (char * count) + '\')')
                elif struct.calcsize(field_struct) == 1:
                    if field_struct == 's':
                        print(f'    {class_var}.m_{field_name}.append(b\'_\')')
                    else:
                        print(f'    {class_var}.m_{field_name}.append(0x01)')
                elif struct.calcsize(field_struct) == 2:
                    print(f'    {class_var}.m_{field_name}.append(0x0102)')
                elif struct.calcsize(field_struct) == 4:
                    print(f'    {class_var}.m_{field_name}.append(0x01020304)')
                elif struct.calcsize(field_struct) == 8:
                    print(f'    {class_var}.m_{field_name}.append(0x0102030405060708)')
            else:
                gen_msg_test_code(field_info)
        if parent_class_is_sub:
            print(f'    # {ref_field_count_var}[].value = len({class_var}.m_{first_field_name})')
        else:
            print(f'    # {ref_field_count_var}.value = len({class_var}.m_{first_field_name})')


if __name__ == '__main__':
    for yamltext in msg_def.yamlTextList:
        data = yaml.load(yamltext, Loader=yaml.Loader)
        msgTreeRoot = {}
        for key, val in data.items():
            process_yaml(key, val, msgTreeRoot, None, None)
        print('''
import struct
import logging
import ctypes
from .Constant import Constant
''')
        for val in msgTreeRoot.values():
            gen_msg_parse_code(val)
            print(f'''
    {'#' * 50}
    # test code
    {'#' * 50}''')
        for val in msgTreeRoot.values():
            gen_msg_test_code(val)
        print('#' * 50)
