import re
from mathqa_exec import parsing
# from parsing import NumericStringParser
from itertools import permutations
from fractions import Fraction

numbers_in_wrods = {"one":1,"two":2,"three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10, "hundred":100, "thousand":1000 }

def refine_number(text):
  
  text = text.replace(",", "").replace(" ", "").replace("×", "*").replace("–","-").replace(":","/").replace('sqrt', '√').replace('sqr', '√').lower().replace('⁄', '/')
  text = text.replace("a)","").replace("b)","").replace("c)","").replace("d)","").replace("e)","").replace("inches","").replace("cubic", "").replace('after', '').replace('m/s', '').replace('msqaure', '').replace('toys', '').replace('yards', '').replace('Â', '').replace('mtrs', '')\
      .replace("dollars", "").replace("percent", '').replace('men', '').replace('grams', '').replace('trousers', '').replace('step/minute', '').replace('gms', '').replace("ways","").replace("way", "").replace("balls","").replace("ball", "").replace("coins","").replace("coin", "").replace("liters", "").replace("-days", "").replace('m/sec', '').replace('days','').replace('day','').replace('hrs', '').replace('hr', '').replace('seconds', '').replace('hour','').replace('sec.', '').replace('sec', '').replace('litres', '').replace('gallons', '').replace('gallon', '')\
      .replace("cm2*square", "").replace("cm²", "").replace("pim^2", "").replace("%", "").replace("cm3","").replace("m3", "").replace("cm2","").replace("m2", "").replace("m³", "").replace('°f', '')\
      .replace('rd', '').replace('nd', '').replace('st', '').replace('am', '').replace('pm', '').replace('mps','').replace('Â', '').replace('°', '').replace("matches", "")\
      .replace("$", "").replace('rs.', "").replace("sqm", "").replace("cum", "").replace("squareinches","").replace("degrees","").replace("kg.","").replace('th', '').replace('mangoes', '')\
      .replace("s.", "").replace("sq.units", "").replace("cc", "").replace("m².", "").replace("gallons", "").replace('kg', '').replace('rs', '')\
      .replace("kilometerssquared", "").replace("increase","").replace("decrease","").replace("minutes","").replace("minute","").replace("hours","").replace("hour","").replace('ways', '')\
      .replace("cu.m","").replace("feet","").replace("sq.cm","").replace("sqcms","").replace("sq.metres","").replace("°","").replace("cmcube","")\
      .replace("m^2","").replace("meters","").replace("deficit","").replace("excess","")\
      .replace("degree","").replace("cm2*square","").replace("squareinch","").replace("years","").replace("rounds","").replace("kg","")\
      .replace("min","").replace("mcube","").replace("mÂ²","").replace("blocks","").replace("ft","").replace("kg","")\
      .replace("miles","").replace("mile","").replace("sqcm","").replace("a.","").replace("b.","").replace("c.","")\
      .replace("d.","").replace("e.","").replace("liters","")\
      .replace("units","").replace("inces","").replace("sq.","")\
      .replace("etres","").replace("sq..","").replace("square","").replace("ete","").replace("cubes","").replace("kg","")\
      .replace("cms", "").replace("cm","").replace("mm", "").replace("km", "").replace("dm", "").replace("mâ²","")\
      .replace("mr", "").replace("ph", "").replace("by", "").replace("less", "").replace("games", "").replace("game", "").replace("ohms", "")\
      .replace("seconds", "").replace("second", "").replace("squnit", "").replace("seedpackets", "").replace("more", "")\
      .replace("lts", "").replace("rise", "").replace("cu", "").replace("percent", "").replace("times", "").replace("yards", "")\
      .replace("are", "").replace("equalto", "").replace("m.", "").replace("additionaledging", "").replace("mtr", "")\
      .replace("tiles", "").replace("mq", "").replace("colours", "").replace("metre", "").replace("rupees", "")\
      .replace("centiqaure", "").replace("ofpetrol", "").replace("sec", "").replace("necklaces", "").replace("sq", "").replace("ts", "")\
      .replace("seedpacket", "").replace("necklace", "").replace("remainder", "").replace("billion", "").replace("million", "").replace('s', '').replace('w', '').replace('a', '').replace('b', '').replace('c', '').replace('d', '').replace('e', '')
  if text.endswith('.') or text.endswith(','):
    text = text[:-1]
            # .replace("","").replace("","").replace("","").replace("","").replace("","").replace("kg","").replace("l","")\
  # if text.endswith("m"):
    
  p = re.compile('[√\.0-9]+m\n')
  # if text.endswith("m") and not p.match(text + '\n'):
  #   print (text)
  if p.match(text + '\n'):
    text = text.replace("m", "")
    # print (text)
  p = re.compile('[√\.0-9]+s\n')
  if p.match(text + '\n'):
    text = text.replace("s", "")
  p = re.compile('[√\.0-9]+h\n')
  if p.match(text + '\n'):
    text = text.replace("h", "")
  p = re.compile('[√\.0-9]+l\n')
  if p.match(text + '\n'):
    text = text.replace("l", "")
  p = re.compile('[√\.0-9]+o\n')
  if p.match(text + '\n'):
    text = text.replace("o", "")
  p = re.compile('[√\.0-9]+c\n')
  if p.match(text + '\n'):
    text = text.replace("c", "")

  # p = re.compile('[0-9]+[a-z]+\n')
  # if p.match(text + '\n'):
  #   print(text)
  return text

def refine_pi(text):
  text = text.replace("∏", "π")
  text = text.replace("(π)","(pi)").replace("√π", "√pi")
  # print (text)
  p = re.compile(".*[0-9]+pi")
  if p.match(text):
    return text.replace("pi", "*pi")
  p = re.compile(".*[0-9]+πr")
  if p.match(text):
    return text.replace("π", "*pi*")
  p = re.compile(".*/π")
  if p.match(text):
    return text.replace("/π", "/pi")
  p = re.compile(".*[0-9]+π")
  if p.match(text):
    return text.replace("π", "*pi")
  p = re.compile(".*[0-9]+[\+∗]π")
  if p.match(text):
    return text.replace("∗π", "*pi")
  p = re.compile(".*\(π.*")
  if p.match(text):
    return text.replace("π", "pi")
  if text.startswith("π"):
    return text.replace("π", "pi*")
  # if "π" in text:

  return text

def refine_multiply(text):
  text = text.replace(")(",")*(")
  for i in range(0,9):
    text = text.replace(str(i)+"(",str(i)+"*(")
  return text

def refine_equality(text):
  p = re.compile("[a-z]=[0-9]+")
  if p.match(text):
    return text[text.index('=')+1:]
  return text

def refine_sqrt(text):
  refine_text = ''
  last_char_seen = ''
  sqrt_seen= False
  paranteses_seen = False
  for char in text:
    if char == "√":
      sqrt_seen = True
      refine_text = refine_text + "sqrt("
      last_char_seen = ''
      continue
    if sqrt_seen == True:
      if char == '(':
        paranteses_seen = True
        continue
      if char == ')':
        if paranteses_seen == True:
          refine_text = refine_text + char
        else:
          refine_text = refine_text + char + ')'
        sqrt_seen = False
        paranteses_seen = False
        last_char_seen = ''
        continue

      elif(not(char >='0' and char <='9') and (last_char_seen >='0' and last_char_seen <= '9')) and paranteses_seen == False:
        refine_text = refine_text + ')' + char 
        sqrt_seen = False
        paranteses_seen = False
        last_char_seen = ''
        continue
      else:
        last_char_seen = char
    refine_text = refine_text + char
  if sqrt_seen == True:
    refine_text = refine_text + ')'

  if 'sqr' in refine_text and 'sqrt' not in refine_text:
    refine_text = refine_text.replace("sqr", "sqrt")
  p = re.compile(".*[0-9]+sqrt.*")
  if p.match(refine_text):
    return refine_text.replace("sqrt", "*sqrt")
  return refine_text


def check_if_number(text):
  number_list = ["one","two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
  if text in number_list:
    return True
  dot_count = 0
  for char in text:
    if char == '.':
      dot_count += 1
  if dot_count > 1:
    return False
  text = text.replace(".", "")
  return text.replace("/","").replace(":","").isdigit() and "²" not in text

def check_non_digit(text):
  for char in text:
    if char >= '0' and char <='9':
      return False
  return True      

def to_float(word):
    if word in numbers_in_wrods:
        return numbers_in_wrods[word]
    if '/' in word:
        word_parts = word.split('/')
        if len(word_parts[0]) == 0 or len(word_parts[1]) == 0:
            word = re.sub("/", "", word)
            word = re.sub("\.", "", word)
            return float(word)
        if int(re.sub("\.", "", word_parts[1])) == 0:
          return 0
        num = float(word_parts[0])/ float(word_parts[1])
        return float(num)
    elif ',' in word:
        word = re.sub(',', '', word)
    return float(word)


def find_non_numeric_values(answer_text):
      nsp = parsing.NumericStringParser()
      answer_text = answer_text.replace("\'", "").replace(" ", "").lower()
      answer_text = answer_text.replace("e)e)", "e)").replace("d)d)", "d)").replace("c)c)", "c)").replace("b)b)", "b)").replace("a)a)", "a)")
      answer_text = answer_text[answer_text.index(")") + 1:]
      if check_if_number(answer_text) == True and ':' not in answer_text and '⁶' not in answer_text:
        return to_float(answer_text)

      answer_text = refine_number(answer_text)
      if (answer_text == '0'):
        return '0'
      if check_if_number(answer_text) == False:
        if answer_text != "noneohese" and answer_text != "none" and answer_text != "cannotbedred" \
            and answer_text != "datainadequate" and answer_text != "noneofabove" and answer_text != "noneoheabove" \
            and "only" not in answer_text and "and" not in answer_text and answer_text != "noneohes" and "&" not in answer_text\
            and "or" not in answer_text: 


          if check_non_digit(answer_text) == True:
            return None
          refine_text = refine_equality(refine_sqrt(refine_multiply(refine_pi(answer_text))))
          try:
            result = nsp.eval(refine_text.replace("−", "-"))
            return float(result)
          except:
            return None
        else:
          return None  
      else:
        return to_float(answer_text)
