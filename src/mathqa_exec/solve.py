from mathqa_exec import new_DataStructure as ds
from mathqa_exec import find_non_numeric_answers as fn
import re
import argparse
import math

########################## Helper function ###################################

numbers_in_wrods = {"one":1,"two":2,"three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10, "hundred":100, "thousand":1000 }

def is_number(word):
    word = re.sub("/", "", word)
    word = re.sub(",", "", word)
    word = re.sub("\.", "", word)
    return word.isdigit() and '²' not in word and '³' not in word and '¹' not in word and ('²' not in word) and ('³' not in word) and ('¹' not in word) and('₂' not in word) and ('⁶' not in word) and ('₃' not in word) and '⁹' not in word and '⁵' not in word and '₁' not in word and '₄' not in word and '⁷' not in word and '⁴' not in word and '⁸' not in word and '₈' not in word

def to_float(word):
    if word in numbers_in_wrods:
        return numbers_in_wrods[word]
    if '/' in word:
        word_parts = word.split('/')
        if len(word_parts[0]) == 0 or len(word_parts[1]) == 0:
            word = re.sub("/", "", word)
            return float(word)
        num = Fraction(int(word_parts[0]), int(word_parts[1]))
        return float(num)
    elif ',' in word:
        word = re.sub(',', '', word)
    if '_' in word:
      word = re.sub('_', '', word)
    return float(word)




######################### Reading and parsing ####################################
def read_src_tgt(test_complete_info_file_name, test_src_file_name, pred_file_name, n_best_check):
  total_set_mapping = {} 
  test_data_src = []
  test_data_tgt = []
  test_data_info = []

 
  total_set_input_file = open(test_complete_info_file_name, encoding="utf8")
  for line in total_set_input_file:
    line_parts = line[:-1].lower().split('\t') #0-problem, 1-rationale 2-formula, 3-correct 4-options
    if len(line_parts) == 9:
      total_set_mapping[line_parts[-2].replace('1 / 4', '0.25').replace('1 / 2', '0.5').replace('1 / 3', '0.33')] = (line_parts[-1], line_parts[3],line_parts[6] ,line_parts[4])
    else:
      total_set_mapping[line_parts[0].replace('1 / 4', '0.25').replace('1 / 2', '0.5').replace('1 / 3', '0.33')] = (line_parts[1], line_parts[3],line_parts[4] ,line_parts[2])

  test_input_src_file = open(test_src_file_name, encoding="utf8")

  for line in test_input_src_file:
    test_data_src.append(line[:-1].lower().replace('1 / 4', '0.25').replace('1 / 2', '0.5').replace('1 / 3', '0.33'))
  
  if type(pred_file_name) == 'str':
    test_input_tgt_file = open(pred_file_name)
    count = 0
    new_test = []
    for line in test_input_tgt_file:
      new_test.append(line[:-1].lower().replace("0_3937", "0.3937").replace('const_0_33', 'const_0.33').replace("const_0_25", 'const_0.25').replace("0_2778", "0.2778").replace("0_6", "0.6").replace("1_6", "1.6").replace("3_6", "3.6").replace("__", " "))
      count +=1 
      if count == n_best_check:
        test_data_tgt.append(new_test)
        new_test= []
        count = 0
    for i in range(len(test_data_src)):
      test_data_info.append(total_set_mapping[test_data_src[i]])
  else:
    count = 0
    new_test = []
    for line in pred_file_name:
      new_test.append(line.lower().replace("0_3937", "0.3937").replace('const_0_33', 'const_0.33').replace("const_0_25", 'const_0.25').replace("0_2778", "0.2778").replace("0_6", "0.6").replace("1_6", "1.6").replace("3_6", "3.6").replace("__", " "))
      count +=1 
      if count == n_best_check:
        test_data_tgt.append(new_test)
        new_test= []
        count = 0
    for i in range(len(test_data_src)):
      test_data_info.append(total_set_mapping[test_data_src[i]])

  return test_data_src, test_data_tgt, test_data_info

def get_src_numbers(test_src_text):
  num_list = []
  test_src_text_words = test_src_text.split(' ')
  for i in range(len(test_src_text_words)):
    word = test_src_text_words[i]
    if is_number(word):
      if i> 0 and test_src_text_words[i-1] == '-':
        num_list.append(to_float(word) * -1)
      else:
        num_list.append(to_float(word))
  return num_list

def beautify(operation_program):
  res_string = ''
  opetaion_list = operation_program.split('__ ')
  for operation in opetaion_list:
    operation_parts = operation.split('__')
    res_string = res_string + operation_parts[0] + '('
    for i in range(1, len(operation_parts)):
      res_string = res_string + operation_parts[i] + ', '
    res_string = res_string[:-2] + ') '
  return res_string[:-1]

def parse_options(options):
  res_opts = []
  options = options.replace('u\'', '').replace('\"', '\'').replace('\'', '').replace(']', '').replace('[','').replace('  ', ', ').split(', ')
  for opt in options:
    if ')' not in opt:
      opt = 'a)' + opt
    res_opts.append(fn.find_non_numeric_values(opt.replace(' ', '')))
  return res_opts

def get_the_sample_score(pred, num_list):
  if len(pred) == 0:
    return 0
  total_score = 0
  count_of_consts = 0
  for i in range(len(pred)):
    if pred[i].startswith('const'):
      count_of_consts += 1
  total_score -= count_of_consts/(len(pred)+0.0)
  for i in range(len(num_list)):
    if ('n' + str(i)) in pred:
      total_score += 2/((i + 1 + 0.0) * (pred.index('n' + str(i)) + 1))
    else:
      total_score -= 1/(i + 1 + 0.0)
  if '#0' in pred:
    total_score += 1.0/(pred.index('#0'))
  if '#1' in pred:
    total_score += 1.0/(pred.index('#1'))

  return total_score

def reranking(pred1, pred2): #if true pred2 is better
  
  if len(pred1) == 0:
    return True

  if len(pred1) > 2 and pred1[1] == pred1[2]:
    return True
  if len(pred2) > 2 and pred2[1] == pred2[2]:
    return False

  const_2_count_pred1 = 0
  const_2_count_pred2 = 0

  for i in range(len(pred1)):
    if pred1[i] == 'const_2' or pred1[i] == 'const_2.0':
      const_2_count_pred1 += 1
  for i in range(len(pred2)):
    if pred2[i] == 'const_2' or pred2[i] == 'const_2.0':
      const_2_count_pred2 += 1
  if const_2_count_pred1 > const_2_count_pred2:
    return True
  elif const_2_count_pred2 > const_2_count_pred1:
    return False

  const_100_count_pred1 = 0
  const_100_count_pred2 = 0
  for i in range(len(pred1)):
    if pred1[i] == 'const_100' or pred1[i] == 'const_100.0':
      const_100_count_pred1 += 1
  for i in range(len(pred2)):
    if pred2[i] == 'const_100' or pred2[i] == 'const_100.0':
      const_100_count_pred2 += 1
  if const_100_count_pred1 < const_100_count_pred2:
    return True
  # elif const_100_count_pred1 > const_100_count_pred2:
  #   return False
  

  # if 'n0' not in pred2:
  #   return False
  const_1_count_pred1 = 0
  const_1_count_pred2 = 0
  
  for i in range(len(pred1)):
    if pred1[i] == 'const_1' or pred1[i] == 'const_1.0':
      const_1_count_pred1 += 1
  for i in range(len(pred2)):
    if pred2[i] == 'const_1' or pred2[i] == 'const_1.0':
      const_1_count_pred2 += 1
  if const_1_count_pred1 > const_1_count_pred2:
    return True
  elif const_1_count_pred1 < const_1_count_pred2:
    return False

  const_3_count_pred2 = 0
  const_3_count_pred1 = 0

  for i in range(len(pred1)):
    if pred1[i] == 'const_3' or pred1[i] == 'const_3.0':
      const_3_count_pred1 += 1
  for i in range(len(pred2)):
    if pred2[i] == 'const_3' or pred2[i] == 'const_3.0':
      const_3_count_pred2 += 1
  if const_3_count_pred1 > const_3_count_pred2:
    return True

  const_10_count_pred2 = 0
  const_10_count_pred1 = 0

  for i in range(len(pred1)):
    if pred1[i] == 'const_10' or pred1[i] == 'const_10.0':
      const_10_count_pred1 += 1
  for i in range(len(pred2)):
    if pred2[i] == 'const_10' or pred2[i] == 'const_10.0':
      const_10_count_pred2 += 1

  if const_10_count_pred1 < const_10_count_pred2:
    return False
  # elif const_3_count_pred1 < const_3_count_pred2:
  #   return False

  const_4_count_pred2 = 0
  const_4_count_pred1 = 0

  for i in range(len(pred1)):
    if pred1[i] == 'const_4' or pred1[i] == 'const_4.0':
      const_4_count_pred1 += 1
  for i in range(len(pred2)):
    if pred2[i] == 'const_4' or pred2[i] == 'const_4.0':
      const_4_count_pred2 += 1
  if const_4_count_pred1 > const_4_count_pred2:
    return True
  elif const_4_count_pred1 < const_4_count_pred2:
    return False

  # if pred1 == ['add', 'n1', 'n2', 'multiply', 'n1', 'n3', 'divide', '#0', '#1']:
  #   import pdb; pdb.set_trace()

  if 'n3' in pred1 and 'n3' not in pred2:
    return False
  if 'n3' in pred2 and 'n3' not in pred1:
    return True
  if 'n2' in pred2 and 'n2' not in pred1:
    return True
  if 'n2' in pred1 and 'n2' not in pred2:
    return False
  if 'n4' in pred1 and 'n4' not in pred2:
    return False
  if 'n4' in pred2 and 'n4' not in pred1:
    return True


  if 'n0' not in pred1 and 'n0' in pred2:
    return True
  if 'n0' in pred1 and 'n0' in pred2:
    if pred1.index('n0') > pred2.index('n0'):
      if 'n1' in pred1 and 'n1' in pred2 or 'n1' not in pred1 and 'n1' not  in pred2 or 'n2' not in pred1 and 'n2' not  in pred2:
        return True
    elif pred1.index('n0') < pred2.index('n0'):
      return False
  
  if 'n1' in pred1 and 'n1' in pred2:
    if pred1.index('n1') > pred2.index('n1'):
      return True
    elif pred1.index('n1') < pred2.index('n1'):
      return False
  if 'n2' in pred1 and 'n2' in pred2:
    if pred1.index('n2') < pred2.index('n2'):
      return True
    elif pred1.index('n2') < pred2.index('n2'):
      return False

  if 'n0' not in pred1:
    return True
  if 'n0' not in pred2:
    return False

  if len(pred2) > len(pred1):
    return True
  elif len(pred2) < len(pred1):
    return False

  if '#0' in pred1 and '#0' in pred2:
    if pred1.index('#0') > pred2.index('#0'):
      return True
    elif pred1.index('#0') < pred2.index('#0'):
      return False
  
  return False




def solve_procedure(test_complete_info_file_name, test_src_file_name, pred_file_name, n_best_check):
  test_data_src, test_data_tgt, test_data_info = read_src_tgt(test_complete_info_file_name, test_src_file_name, pred_file_name, n_best_check)
  score_count = 0
  found_solution = []

  for i in range(0,len(test_data_src)):
    best_ii = -1
    best_candidate_list = []
    best_difference = 1000
    best_inst = ''
    number_list = get_src_numbers(test_data_src[i])
    options_values = parse_options(list(test_data_info[i])[2])
    max_op = 0
    ans_th = [0.1, 1]
    for l in range(len(options_values)):
      if options_values[l] !=None and max_op < float(options_values[l]):
        max_op = options_values[l]
    if max_op != 0:
      ans_th.append(min((float(max_op)/4)+1, 10))

    for ii in range(0, n_best_check):
      for ans in ans_th:
        prediction_words = test_data_tgt[i][ii].replace('  ', ' ').split(' ')
        temp_memory = []
        current_inst = ''
        used_num_flg = False
        used_const_flag = False
        for prediction_word in prediction_words:
          if prediction_word in ds.operation_dictionary_structure.operation_names:
            if current_inst != '':
              ret_value = current_inst.execute()
              if ret_value != None:
                temp_memory.append(ret_value)
            current_inst = ds.Instruction(prediction_word)
          else:
            if current_inst == '':
              break
            if prediction_word == 'const_pi':
              current_inst.add_arguemnt(3.1415)
              used_const_flag = True
            elif prediction_word == 'const_deg_to_rad':
              current_inst.add_arguemnt(0.0055)
              used_const_flag = True
            elif prediction_word.startswith('const_'):
              used_const_flag = True
              current_inst.add_arguemnt(float(prediction_word[6:]))
            elif prediction_word.startswith('#'):
              if int(prediction_word[1:]) < len(temp_memory):
                current_inst.add_arguemnt(temp_memory[int(prediction_word[1:])])
            elif prediction_word.startswith('n'):
              used_num_flg = True
              if int(prediction_word[1:]) < len(number_list):
                current_inst.add_arguemnt(number_list[int(prediction_word[1:])])
            elif prediction_word != '':
              current_inst.add_arguemnt(float(prediction_word))
        if current_inst == '':
          ans_found = False
          continue
        res = current_inst.execute()
        if res != None:
          temp_memory.append(res)
        jj = len(temp_memory) -1
        

        if jj < 0 or used_num_flg==False:
          continue
        ans_candidates = []
        min_distance = 1000
        

        for j in range(len(options_values)):
            if options_values[j]!= None \
              and abs(float(temp_memory[jj]) - float(options_values[j])) < ans:
              if (min_distance > abs(float(temp_memory[jj]) - float(options_values[j]))):
                min_distance = abs(float(temp_memory[jj]) - float(options_values[j]))
              ans_candidates.append(j)

        if ans_candidates != [] and ((min_distance < best_difference and \
                (len(ans_candidates) < 3 or ans_candidates == []) and\
                (ii-best_ii < 20 or best_ii == -1) and \
                (len(best_candidate_list)== 0 or len(best_candidate_list) >= len(ans_candidates)) \
                )): 

            best_candidate_list = ans_candidates
            best_difference = min_distance
            best_ii = ii
            best_inst = prediction_words

    

  
    if best_candidate_list == []: # if it is the last one and we have not found an answer.
      if i in found_solution:
        import pdb; pdb.set_trace()
      found_solution.append(i)
      score_count += 0.2
        
      # print('chance for problem ' + str(i))
    elif int(ord(list(test_data_info[i])[1])-ord('a')) in best_candidate_list:
      if i in found_solution:
        import pdb; pdb.set_trace()
      found_solution.append(i)
      # print("for index. "+ str(i) +". adding:" + str( 1.0 / (len(best_candidate_list)))) 
      score_count += 1.0 / (len(best_candidate_list))
    
 
  print("accuracy would be: "+ str((score_count+0.0) /len(test_data_src)))
  return (score_count+0.0) /len(test_data_src)


def main():


    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src_file_name", default="data_set/allcomplete_src_test.txt", type=str, help="The file containing the source sentences")
    parser.add_argument("-info_file_name", default="data_set/complete_all_for_release.tsv", type=str, help="The file containing the infromation of the problem.")
    parser.add_argument("--predictions", default="pred_all_separate_adam_new_adam.txt", type=str, help="The path to the file containing the prediction paths")
    parser.add_argument("--n_best", default=100, type=int, help="Number of predictions per problem.")
    args = parser.parse_args()

    solve_procedure(args.info_file_name, args.src_file_name, args.predictions, args.n_best)



if __name__ == "__main__":
    main()
