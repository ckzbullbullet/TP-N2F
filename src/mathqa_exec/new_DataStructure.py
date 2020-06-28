import re
import math
from itertools import permutations
from fractions import Fraction
import gc
import fractions

numbers_in_wrods = {"one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10, "hundred":100, "thousand":1000 }
prime_number_list = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999,2003,2011,2017,2027,2029,2039,2053,2063,2069,2081,2083,2087,2089,2099,2111,2113,2129,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213,2221,2237,2239,2243,2251,2267,2269,2273,2281,2287,2293,2297,2309,2311,2333,2339,2341,2347,2351,2357,2371,2377,2381,2383,2389,2393,2399,2411,2417,2423,2437,2441,2447,2459,2467,2473,2477,2503,2521,2531,2539,2543,2549,2551,2557,2579,2591,2593,2609,2617,2621,2633,2647,2657,2659,2663,2671,2677,2683,2687,2689,2693,2699,2707,2711,2713,2719,2729,2731,2741,2749,2753,2767,2777,2789,2791,2797,2801,2803,2819,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903,2909,2917,2927,2939,2953,2957,2963,2969,2971,2999,3001,3011,3019,3023,3037,3041,3049,3061,3067,3079,3083,3089,3109,3119,3121,3137,3163,3167,3169,3181,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257,3259,3271,3299,3301,3307,3313,3319,3323,3329,3331,3343,3347,3359,3361,3371,3373,3389,3391,3407,3413,3433,3449,3457,3461,3463,3467,3469,3491,3499,3511,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571,3581,3583,3593,3607,3613,3617,3623,3631,3637,3643,3659,3671,3673,3677,3691,3697,3701,3709,3719,3727,3733,3739,3761,3767,3769,3779,3793,3797,3803,3821,3823,3833,3847,3851,3853,3863,3877,3881,3889,3907,3911,3917,3919,3923,3929,3931,3943,3947,3967,3989,4001,4003,4007,4013,4019,4021,4027,4049,4051,4057,4073,4079,4091,4093,4099,4111,4127,4129,4133,4139,4153,4157,4159,4177,4201,4211,4217,4219,4229,4231,4241,4243,4253,4259,4261,4271,4273,4283,4289,4297,4327,4337,4339,4349,4357,4363,4373,4391,4397,4409,4421,4423,4441,4447,4451,4457,4463,4481,4483,4493,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583,4591,4597,4603,4621,4637,4639,4643,4649,4651,4657,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751,4759,4783,4787,4789,4793,4799,4801,4813,4817,4831,4861,4871,4877,4889,4903,4909,4919,4931,4933,4937,4943,4951,4957,4967,4969,4973,4987,4993,4999,5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443,5449,5471,5477,5479,5483,5501,5503,5507,5519,5521,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791,5801,5807,5813,5821,5827,5839,5843,5849,5851,5857,5861,5867,5869,5879,5881,5897,5903,5923,5927,5939,5953,5981,5987,6007,6011,6029,6037,6043,6047,6053,6067,6073,6079,6089,6091,6101,6113,6121,6131,6133,6143,6151,6163,6173,6197,6199,6203,6211,6217,6221,6229,6247,6257,6263,6269,6271,6277,6287,6299,6301,6311,6317,6323,6329,6337,6343,6353,6359,6361,6367,6373,6379,6389,6397,6421,6427,6449,6451,6469,6473,6481,6491,6521,6529,6547,6551,6553,6563,6569,6571,6577,6581,6599,6607,6619,6637,6653,6659,6661,6673,6679,6689,6691,6701,6703,6709,6719,6733,6737,6761,6763,6779,6781,6791,6793,6803,6823,6827,6829,6833,6841,6857,6863,6869,6871,6883,6899,6907,6911,6917,6947,6949,6959,6961,6967,6971,6977,6983,6991,6997,7001,7013,7019,7027,7039,7043,7057,7069,7079,7103,7109,7121,7127,7129,7151,7159,7177,7187,7193,7207,7211,7213,7219,7229,7237,7243,7247,7253,7283,7297,7307,7309,7321,7331,7333,7349,7351,7369,7393,7411,7417,7433,7451,7457,7459,7477,7481,7487,7489,7499,7507,7517,7523,7529,7537,7541,7547,7549,7559,7561,7573,7577,7583,7589,7591,7603,7607,7621,7639,7643,7649,7669,7673,7681,7687,7691,7699,7703,7717,7723,7727,7741,7753,7757,7759,7789,7793,7817,7823,7829,7841,7853,7867,7873,7877,7879,7883,7901,7907,7919]

def check_redundant_operation(operation_name, arg_list):
  if operation_name in ['add', 'sqrt', 'circle_area', 'circumface', 'square_perimeter', 'volume_cube', 'twice', 'triple']:
    if 0 in arg_list:
      return True
  if operation_name in ['subtract']:
    if arg_list[1] == 0:
      return True
  if operation_name in ['multiply', 'power', 'sqrt', 'factorial', 'volume_cube']:
    if 1 in arg_list:
      return True
  if operation_name in ['divide']:
    if arg_list[1] == 1:
      return True

  return False

def check_unity_arg_list(op1, op2):
  arg_flg = False
  for arg in op1.argument_list:
    if arg not in op2.argument_list:
      arg_flg = True
      break
  if arg_flg == True:
    arg_flg = False
    for arg in op2.argument_list:
      if arg not in op1.argument_list:
        arg_flg = True
        break
  if arg_flg == True:
    return False
  return True

def operation_equal(op1, op2):
  res1 = op1.execute()
  res2 = op2.execute()
  if res1 == res2 and res1 != None:
    return True
  else:
    return False

def is_number(word):
  if word in numbers_in_wrods:
    return True
  word = re.sub("\/", "", word)
  word = re.sub(",", "", word)
  word = re.sub("\.", "", word)
  return word.isdigit()

def is_fraction(word):
  return '/' in word and is_number(word)

def is_thausand(word):
  return ',' in word and is_number(word)

def is_float(word):
  return '.' in word and is_number(word)

def change_to_float(word):
  try:
    if word in numbers_in_wrods:
      return numbers_in_wrods[word]
    if '/' in word:
      word_parts = word.split('/')
      num = Fraction(int(word_parts[0]), int(word_parts[1]))
      return float(num)
    elif ',' in word:
      word = re.sub(',', '', word)
    return float(word)
    if '_' in word:
      word = re.sub(',', '', word)
  except:
    return word


class operation_dictionary_structure:
    operation_names = ['add', 'subtract', 'multiply', 'divide', \
              'log', 'sqrt', 'factorial', 'gcd', 'lcm', 'power',  'max', 'min', 'remindar', 'reminder', 'negate', 'inverse', 'round', 'floor',\
              'sine', 'cosine', 'tangent', 'radians_to_degree', 'degree_to_radians',\
              'sum_consecutive_number',\
              #geometry
              'circle_area', 'circumface', 'circle_arc', 'semi_circle_perimiter', 'circle_sector_area', \
              'rectangle_perimeter', 'rectangle_area', 'square_perimeter', 'square_area', 'trapezium_area', 'rhombus_perimeter', 'rhombus_area','quadrilateral_area',\
              'volume_cone', 'volume_rectangular_prism', 'volume_cube', 'volume_sphere', 'volume_cylinder', \
              'surface_cone', 'surface_cylinder', 'surface_cube', 'surface_rectangular_prism', 'surface_sphere', \
              'side_by_diagonal',  'cube_edge_by_volume', 'diagonal', 'square_edge_by_perimeter', 'square_edge_by_parameter','square_edge_by_area',\
              'triangle_perimeter', 'triangle_area','triangle_area_three_edges', \
              #probability
              'union_prob', 'negate_prob', 'choose', 'permutation', 'count_interval',\
              #gain
              'percent', 'p_after_gain', 'p_after_loss'\
              'price_after_gain','price_after_loss',\
              'from_percent', 'gain_percent', 'loss_percent', 'negate_percent',\
              'original_price_before_gain', 'original_price_before_loss', 'to_percent',\
              #physics
              'speed', 'combined_work', 'find_work',\
              'speed_ratio_steel_to_stream','speed_in_still_water', 'stream_speed']

           
            
    @staticmethod
    def get_operation_memory_output_mapping(operatione_name):
      if operatione_name not in operation_dictionary_structure.operation_memory_output_mapping:
        return None
      return operation_dictionary_structure.operation_memory_output_mapping[operatione_name]

    @staticmethod
    def find_matching_operation_by_category(category_list):
      candidate_operation_list = []
      for i in range(len(category_list)):
        # import pdb; pdb.set_trace()
        opertation_in_category = operation_dictionary_structure.problem_categories[int(category_list[i][0])]
        candidate_operation_list.append([])
        for op in opertation_in_category:
          candidate_operation_list[i].append(op)
        for op in operation_dictionary_structure.general_operation_names:
          if op not in candidate_operation_list[i]:
            candidate_operation_list[i].append(op)

      return candidate_operation_list

class Instruction:
    def __init__(self, name, argument_list=[]):
        self.name = name
        self.argument_list = []
        self.ret_value = ''

    def add_arguemnt(self, argument):
        if 'str' in str(type(argument)):
          argument = re.sub(",", "", argument)
        self.argument_list.append(argument)

    def execute(self):
        ret_value = None
        try:
          if self.name == 'add':
              ret_value = float(self.argument_list[0]) + float(self.argument_list[1])
          elif self.name == 'subtract':
            ret_value = float(self.argument_list[0]) - float(self.argument_list[1])
          elif self.name == 'multiply': 
            ret_value = float(self.argument_list[0]) * float(self.argument_list[1])
          elif self.name == 'divide':
            if float(self.argument_list[1]) != 0 and abs(float(self.argument_list[1])) > 0.001:
              ret_value = float(self.argument_list[0]) / float(self.argument_list[1])
          elif self.name == 'log':
            if float(self.argument_list[0]) > 0:
              ret_value = math.log(float(self.argument_list[0]) , 2)
          elif self.name == 'sqrt':
            if float(self.argument_list[0]) >= 0:
              ret_value = math.sqrt(float(self.argument_list[0]))
          elif self.name == 'factorial':
            if float(self.argument_list[0]) < 100:
              ret_value = math.factorial(int(self.argument_list[0]))
          elif self.name == 'gcd':
            res = fractions.gcd(float(self.argument_list[0]), float(self.argument_list[1]))
            if res == 1:
              ret_value == None
            else:
              ret_value = res
          elif self.name == 'lcm':
            gcd = fractions.gcd(float(self.argument_list[0]), float(self.argument_list[1]))
            ret_value = (float(self.argument_list[0])*float(self.argument_list[1]))/(gcd + 0.0)
          elif self.name == 'power':
            if float(self.argument_list[1]) < 10 and float(self.argument_list[0]) > 0 and float(self.argument_list[1]) > 0:
              ret_value = float(self.argument_list[0]) ** float(self.argument_list[1])
          elif self.name == 'max':
            ret_value = max(float(self.argument_list[0]), float(self. argument_list[1]))
          elif self.name == 'min':
            ret_value = min(float(self.argument_list[0]), float(self. argument_list[1]))
          elif self.name == 'remindar' or self.name == 'reminder':
            if float(self.argument_list[1])!= 0:
              ret_value = float(self.argument_list[0])% float(self.argument_list[1])
          elif self.name == 'negate':
            ret_value = float(self.argument_list[0]) * (-1)
          elif self.name == 'inverse':
            if float(self.argument_list[0]) !=0:
              ret_value = (1.0) / float(self.argument_list[0])
          elif self.name  == 'round':
              if (float(self.argument_list[0]) * 1000) % 10 > 4:
                ret_value = float("{0:.4f}".format(float(self.argument_list[0]))) + 0.0001
              else:
                ret_value = float("{0:.4f}".format(float(self.argument_list[0])))
          elif self.name == 'floor':
            ret_value = math.floor(float(self.argument_list[0]))
          elif self.name == 'sine':
            if float(self.argument_list[0]) <=1:
              ret_value = math.asin(float(self.argument_list[0]))
          elif self.name == 'cosine':
            if float(self.argument_list[0]) <=1:               
              ret_value = math.acos(float(self.argument_list[0]))
          elif self.name == 'tangent':
            ret_value = math.atan(float(self.argument_list[0]))
          elif self.name == 'radians_to_degree':
            ret_value = float(self.argument_list[0]) * 180
          elif self.name == 'radians_to_degree':
            ret_value = float(self.argument_list[0]) / 180
          elif self.name == 'sum_consecutive_number':
            ret_value=0 
            for num in range(int(self.argument_list[0]), int(self.argument_list[1])):
              ret_value += num
          elif self.name == 'circle_area':
              ret_value = float(self.argument_list[0])*float(self.argument_list[0])*math.pi
          elif self.name == 'circumface':
              ret_value = float(self.argument_list[0]) * 2 * math.pi
          elif self.name == 'circle_arc':
            ret_value = (float(self.argument_list[0])/360.0) * math.pi * 2 * float(self.argument_list[1])
          elif self.name == 'semi_circle_perimiter':
            ret_value = math.pi * float(self.argument_list[0]) + (2*float(self.argument_list[0]))
          elif self.name == 'circle_sector_area':
            if float(self.argument_list[1]) <= 360 and float(self.argument_list[1]) >= 0:
              ret_value = ((float(self.argument_list[1]))/360.0) * math.pi * float(self.argument_list[0]) * float(self.argument_list[0])
          elif self.name == 'rectangle_perimeter':
              ret_value = (float(self.argument_list[0]) + float(self.argument_list[1])) * 2
          elif self.name == 'rectangle_area':
              ret_value = float(self.argument_list[0]) * float(self.argument_list[1])
          elif self.name  == 'square_area':
              ret_value = float(self.argument_list[0]) * float(self.argument_list[0])
          elif self.name == 'square_perimeter':
              ret_value = float(self.argument_list[0]) * 4
          elif self.name == 'trapezium_area':
            ret_value = 0.5 * float(self.argument_list[0]) * (float(self.argument_list[1]) + float(self.argument_list[2]))
          elif self.name == 'rhombus_area':
            ret_value = (float(self.argument_list[0]) * float(self.argument_list[1]))/2
          elif self.name == 'rhombus_perimeter':
            ret_value = float(self.argument_list[0]) * 4
          elif self.name == 'quadrilateral_area':
            ret_value = 0.5 * float(self.argument_list[0]) * (float(self.argument_list[1]) + float(self.argument_list[2]))
          elif self.name == 'volume_cone':
              ret_value = (float(self.argument_list[0]) * float(self.argument_list[0]) * float(self.argument_list[1]) * math.pi)/3
          elif self.name == 'volume_rectangular_prism':
              ret_value = float(self.argument_list[0]) * float(self.argument_list[1]) * float(self.argument_list[2])
          elif self.name == 'volume_cube':
              ret_value = float(self.argument_list[0]) * float(self.argument_list[0]) * float(self.argument_list[0])
          elif self.name == "volume_sphere":
            ret_value = (4.0/3) * math.pi * self.argument_list[0] * self.argument_list[0] * self.argument_list[0]
          elif self.name == 'volume_cylinder':
              ret_value = math.pi * float(self.argument_list[0]) * float(self.argument_list[0]) * float(self.argument_list[1])
          elif self.name == 'surface_cone':
              ret_value = (math.pi * float(self.argument_list[0])) * (float(self.argument_list[0]) + \
                math.sqrt((float(self.argument_list[0])*float(self.argument_list[0])) + (float(self.argument_list[1]) * float(self.argument_list[1]))))
          elif self.name == 'surface_cylinder':
              ret_value = 2 * math.pi * (float(self.argument_list[0])* float(self.argument_list[1]) + \
                float(self.argument_list[0]) * float(self.argument_list[0]))
          elif self.name == 'surface_cube':
              ret_value = float(self.argument_list[0]) * float(self.argument_list[0]) * 6 
          elif self.name == 'surface_rectangular_prism':
              ret_value = 2 *((float(self.argument_list[0]) * float(self.argument_list[1])) +\
               (float(self.argument_list[1]) * float(self.argument_list[2])) +\
               (float(self.argument_list[0]) * float(self.argument_list[2])) )
          elif self.name == "surface_sphere":
            ret_value = 4 * math.pi * self.argument_list[0] * self.argument_list[0]
          elif self.name == 'side_by_diagonal':
            if float(self.argument_list[0])**2 - float(self.argument_list[1])**2 >= 0:
              ret_value = math.sqrt(float(self.argument_list[0])**2 - float(self.argument_list[1])**2)
          elif self.name == 'cube_edge_by_volume':
            if float(self.argument_list[0]) >= 0:
              ret_value = float(self.argument_list[0]) **(0.33333)
          elif self.name == 'diagonal':
            ret_value = math.sqrt(float(self.argument_list[0])**2 + float(self.argument_list[1])**2)
          elif self.name == 'square_edge_by_perimeter':
            ret_value = float(self.argument_list[0])/4
          elif self.name == 'square_edge_by_parameter':
            ret_value = float(self.argument_list[0])/4
          elif self.name == 'square_edge_by_area':
            ret_value = math.sqrt(float(self.argument_list[0]))
          elif self.name == 'triangle_perimeter':
            ret_value = float(self.argument_list[0]) + float(self.argument_list[1]) + float(self.argument_list[2])
          elif self.name == 'triangle_area':
            ret_value = (float(self.argument_list[0]) * float(self.argument_list[1]))/2.0
          elif self.name == 'triangle_area_three_edges':
            max_index = 0
            for ii in range(1, len(self.argument_list)):
              if self.argument_list[ii] > self.argument_list[max_index]:
                max_index = ii
            others_sum = 0
            for ii in range(0, len(self.argument_list)):
              if ii != max_index:
                others_sum += self.argument_list[ii]
            if others_sum > self.argument_list[max_index]:
              p = (float(self.argument_list[0]) + float(self.argument_list[1]) + float(self.argument_list[2]))/2
              ret_value = math.sqrt(p * (p - float(self.argument_list[0]))* (p - float(self.argument_list[1]))* (p - float(self.argument_list[2])))
          elif self.name == 'union_prob':
            if float(self.argument_list[0]) <= 1 and float(self.argument_list[0]) >= 0 and float(self.argument_list[1]) <= 1 and float(self.argument_list[1]) >= 0:
              ret_value = float(self.argument_list[0]) + float(self.argument_list[1]) - float(self.argument_list[2])
          elif self.name == 'negate_prob':
            if float(self.argument_list[0]) <= 1 and float(self.argument_list[0]) >= 0:
              ret_value = 1-float(self.argument_list[0])
          elif self.name == 'choose':  
            self.argument_list[0] = float(self.argument_list[0])
            self.argument_list[1] = float(self.argument_list[1])
            if int(self.argument_list[0]) < 100 and int(self.argument_list[1]) < 100 and int(self.argument_list[1])>0:
              if int(self.argument_list[0]) > int(self.argument_list[1]):
                ret_value = math.factorial(int((self.argument_list[0])))/(math.factorial(int(self.argument_list[1])))* (math.factorial(int(self.argument_list[0]) - int(self.argument_list[1])))
              else:
                ret_value = math.factorial(int((self.argument_list[1])))/(math.factorial(int(self.argument_list[0])))* (math.factorial(int(self.argument_list[1]) - int(self.argument_list[0])))
          elif self.name == 'permutation':
            if float(self.argument_list[1]) - float(self.argument_list[0]) > 0 and self.argument_list[1] < 20:
              ret_value = math.factorial(int(self.argument_list[1]))/ math.factorial(float(self.argument_list[1]) - float(self.argument_list[0]))
            elif self.argument_list[0] < 20:
              ret_value = math.factorial(int(self.argument_list[0]))/ math.factorial(float(self.argument_list[0]) - float(self.argument_list[1]))
          elif self.name == 'count_interval':
            ret_value = float(self.argument_list[0]) - float(self.argument_list[1]) + 1
          elif self.name == 'percent':
            if float(self.argument_list[0]) <=  100 and float(self.argument_list[0]) >= 0:
              ret_value = ((float(self.argument_list[0]))/100) * float(self.argument_list[1])
          elif self.name == 'price_after_gain' or self.name == 'p_after_gain':
              ret_value = (1 + (float(self.argument_list[0])/100)) * float(self.argument_list[1])
          elif self.name == 'price_after_loss' or self.name == 'p_after_loss':
              ret_value = (1 - (float(self.argument_list[0])/100)) * float(self.argument_list[1])
          elif self.name == 'from_percent':
            ret_value = float(self.argument_list[0])/ 100
          elif self.name == 'gain_percent':
            ret_value = 100 + float(self.argument_list[0])
          elif self.name == 'loss_percent':
            ret_value = 100 - float(self.argument_list[0])
          elif self.name == 'negate_percent':
            ret_value = 100-float(self.argument_list[0])
          elif self.name == 'original_price_before_loss':
            ret_value =(float(self.argument_list[1]) * 100.0)/(100.00001 - float(self.argument_list[0]))
          elif self.name == 'original_price_before_gain':
            ret_value =(float(self.argument_list[1]) * 100.0)/(100 + float(self.argument_list[0]))
          elif self.name == 'to_percent':
            ret_value = float(self.argument_list[0])* 100
          elif self.name == 'speed':
              ret_value = float(self.argument_list[0]) / float(self.argument_list[1])
          elif self.name == 'combined_work':
            if self.argument_list[0] > 1:
              self.argument_list[0] = 1.0/(float(self.argument_list[0])+ 0.0)
            if self.argument_list[1] > 1:
              self.argument_list[1] = 1.0/(float(self.argument_list[1])+ 0.0)
            ret_value = 1.0/ (float(self.argument_list[0]) + float(self.argument_list[1]))
          elif self.name == 'find_work':
            if self.argument_list[0] > 1:
              self.argument_list[0] = 1.0/(float(self.argument_list[0])+ 0.0)
            if self.argument_list[1] > 1:
              self.argument_list[1] = 1.0/(float(self.argument_list[1])+ 0.0)
            ret_value = 1.0/ (max(float(self.argument_list[0]), float(self.argument_list[1])) - min(float(self.argument_list[0]), float(self.argument_list[1])))
          elif self.name == 'speed_ratio_steel_to_stream':
            ret_value = (float(self.argument_list[0]) + float(self.argument_list[1])) / (float(self.argument_list[0]) - float(self.argument_list[1]))
          elif self.name == 'speed_in_still_water': # first argument is upstream speed and second argument is downstream speed
            ret_value = (float(self.argument_list[0]) + float(self.argument_list[1])) / 2
          elif self.name == 'stream_speed':
            ret_value = (float(self.argument_list[0]) - float(self.argument_list[1])) / 2



          #######################################################
        except:
          return None

        self.ret_value = ret_value
        return ret_value
