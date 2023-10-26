import re
import random
import unidecode

class Eliza():
  '''
    Class for an E.L.I.Z.A chatbot

    Input:
     - lang: language 
  '''
  def __init__(self, lang="en"):
    self.lang = lang
    if lang=="en":
      from eliza.en import data
    elif lang=="es":
      from eliza.es import data
    else:
      raise Exception("Not implement yet!")

    self.vocabulary = data.reflections
    self.vocabulary_keys = self.vocabulary.keys()

    self.patterns = list(map(lambda x: re.compile(x[0], re.IGNORECASE),data.patts))
    self.responses = list(map(lambda x: x[1],data.patts))

  def translate(self, text):
    words = text.lower().strip().split()
    newwords = [self.vocabulary[word] 
                if word in self.vocabulary_keys else word
                  for word in words
                  ]
    return " ".join(newwords)
  
  def clean(self,text):
    newtext = re.sub(r"[^a-zA-Z]*$","",text)
    return newtext

  def respond(self, text):
    utext = unidecode.unidecode(text)
    for patt,answers in zip(self.patterns,self.responses):
      match = patt.match(utext)
      if match:
        text = self.clean(text)

        #Random choice 
        ans = random.choice(answers)
        
        var_subs = re.findall(r"%[0-9][0-9]*",ans)
        if len(var_subs)>1:
          var_subs = list(set(var_subs))
        n_subs = [int(v[1:]) for v in var_subs]
        
        for n,v in zip(n_subs,var_subs):
          ans = re.sub(v,self.translate(match.group(n)),ans)
        return ans
    return text


