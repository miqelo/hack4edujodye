
#----------------------------------------------------------------------
# gReflections, a translation table used to convert things you say
#    into things the computer says back, e.g. "I am" --> "you are"
#----------------------------------------------------------------------
reflections = {
  "soy"   : "eres",
  "era"  : "eras",
  "fui" : "fuiste",
  "yo"    : "tu",
  "querria"  : "querrias",
  "me gustaria"  : "te gustaria",
  "yo he"  : "tu has",
  "i'll"  : "you will",
  "mi"  : "tu",
  "eres"  : "soy",
  "tu has": "yo he",
  "you'll": "I will",
  "tu"  : "mi",
  "tuyo"  : "mio",
  "mi"  : "tu"
}



#----------------------------------------------------------------------
# gPats, the main response table.  Each element of the list is a
#  two-element list; the first is a regexp, and the second is a
#  list of possible responses, with group-macros labelled as
#  %1, %2, etc.
#----------------------------------------------------------------------
patts = [
  [r'Necesito (.*)',
  [  "¿Por qué necesitas %1?",
    "¿Te ayudaría realmente conseguir %1?",
    "¿Estás seguro de que necesitas %1?"]],

  [r'Por que (tu)? no ([^\?]*)\??',
  [  "¿De verdad crees que yo no %1?",
    "Quizás, en algún momento yo consiga %1.",
    "¿De verdad quieres que yo %1?"]],

  [r'Why can\'?t I ([^\?]*)\??',
  [  "Do you think you should be able to %1?",
    "If you could %1, what would you do?",
    "I don't know -- why can't you %1?",
    "Have you really tried?"]],

  [r'I can\'?t (.*)',
  [  "How do you know you can't %1?",
    "Perhaps you could %1 if you tried.",
    "What would it take for you to %1?"]],

  [r'I am (.*)',
  [  "Did you come to me because you are %1?",
    "How long have you been %1?",
    "How do you feel about being %1?"]],

  [r'I\'?m (.*)',
  [  "How does being %1 make you feel?",
    "Do you enjoy being %1?",
    "Why do you tell me you're %1?",
    "Why do you think you're %1?"]],

  [r'Are you ([^\?]*)\??',
  [  "Why does it matter whether I am %1?",
    "Would you prefer it if I were not %1?",
    "Perhaps you believe I am %1.",
    "I may be %1 -- what do you think?"]],

  [r'What (.*)',
  [  "Why do you ask?",
    "How would an answer to that help you?",
    "What do you think?"]],

  [r'How (.*)',
  [  "How do you suppose?",
    "Perhaps you can answer your own question.",
    "What is it you're really asking?"]],

  [r'porque (.*)',
  [  "¿Es eso un motivo de verdad?",
    "¿Se te ocurre alguna otra razón?",
    "¿Puede explicar ese motivo algo más?",
    "Si %1, ¿qué otra cosa ha de ser cierta?"]],

  [r'(.*) lo siento(.*)',
  [  "Hay muchas ocasiones en las que una disculpa no es necesaria.",
    "¿Qué sentimientos tienes cuando te disculpas?"]],

  [r'hola(.*)',
  [  "Hola... Me alegro de que pudieras acercarte hoy.",
    "Hola... como estás hoy?",
    "Hola, ¿cómo te sientes hoy?"]],

  [r'creo (.*)',
  [  "Do you doubt %1?",
    "Do you really think so?",
    "But you're not sure %1?"]],

  [r'(.*) friend (.*)',
  [  "Tell me more about your friends.",
    "When you think of a friend, what comes to mind?",
    "Why don't you tell me about a childhood friend?"]],

  [r'^Si$',
  [  "Pareces muy seguro.",
    "OK, pero ¿puedes explicarte un poco?"]],

  [r'(.*) computer(.*)',
  [  "Are you really talking about me?",
    "Does it seem strange to talk to a computer?",
    "How do computers make you feel?",
    "Do you feel threatened by computers?"]],

  [r'Is it (.*)',
  [  "Do you think it is %1?",
    "Perhaps it's %1 -- what do you think?",
    "If it were %1, what would you do?",
    "It could well be that %1."]],

  [r'It is (.*)',
  [  "You seem very certain.",
    "If I told you that it probably isn't %1, what would you feel?"]],

  [r'Can you ([^\?]*)\??',
  [  "What makes you think I can't %1?",
    "If I could %1, then what?",
    "Why do you ask if I can %1?"]],

  [r'Can I ([^\?]*)\??',
  [  "Perhaps you don't want to %1.",
    "Do you want to be able to %1?",
    "If you could %1, would you?"]],

  [r'You are (.*)',
  [  "Why do you think I am %1?",
    "Does it please you to think that I'm %1?",
    "Perhaps you would like me to be %1.",
    "Perhaps you're really talking about yourself?"]],

  [r'You\'?re (.*)',
  [  "Why do you say I am %1?",
    "Why do you think I am %1?",
    "Are we talking about you, or me?"]],

  [r'I don\'?t (.*)',
  [  "Don't you really %1?",
    "Why don't you %1?",
    "Do you want to %1?"]],

  [r'I feel (.*)',
  [  "Good, tell me more about these feelings.",
    "Do you often feel %1?",
    "When do you usually feel %1?",
    "When you feel %1, what do you do?"]],
  [r'he (.*)',
  [  "¿Por qué dices que has %1?",
    "¿De verdad has %1?",
    "Ahora que has %1, ¿qué será lo siguiente que hagas?"]],
  [r'tengo que (.*)',
  [  "¿Por qué dices que tienes que %1?",
    "¿De verdad tienes que %1?",
    "Ahora que sabes que tienes que %1, ¿qué es lo siguientes que harás?"]],

  [r'me gustaria (.*)',
  [  "Puedes explicar por qué te gustaría %1?",
    "¿Por qué querrías %1?",
    "¿Quién más sabe que tu querrías %1?"]], 
  [r'querria (.*)',
  [  "Puedes explicar por qué te gustaría %1?",
    "¿Por qué querrías %1?",
    "¿Quién más sabe que tu querrías %1?"]],

  [r'Hay (.*)',
  [  "¿Tú crees que hay %1?",
    "Es probable que haya %1.",
    "¿Te gustaría que hubiera o hubiese %1?"]],

  [r'Mi (.*)',
  [  "Entiendo, tu %1.",
    "¿Por qué dices tu %1?",
    "Cuando tu %1, ¿cómo te sientes?"]],

  [r'tu (.*)',
  [  "Deberíamos estar hablando sobre tí, no sobre mi.",
    "¿Por qué dices eso sobre mi?",
    "¿Por qué te importa si yo %1?"]],

  [r'Why (.*)',
  [  "Why don't you tell me the reason why %1?",
    "Why do you think %1?" ]],

  [r'quiero (.*)',
  [  "¿Qué significaría para ti si consiguieras %1?",
    "¿Por qué quieres %1?",
    "¿Qué harías si consiguieras %1?",
    "Si consiguieres %1, entonces ¿qué harías?"]],

  [r'(.*) madre(.*)',
  [  "Cuéntame más sobre tu madre.",
    "¿Cómo era tu relación con tu madre?",
    "¿Cómo te hacía sentir tu madre?",
    "¿Cómo te sientes al hablar de tu madre?",
    "¿La relación con tu madre afecta a tus sentimientos hoy?",
    "Que relaciones familiares sean buenas es importante."]],

  [r'(.*) padre(.*)',
  [ "Cuéntame más sobre tu padre.",
    "¿Cómo era tu relación con tu padre?",
    "¿Cómo te hacía sentir tu padre?",
    "¿Cómo te sientes al hablar de tu padre?",
    "¿La relación con tu padre afecta a tus sentimientos hoy?",
    "Que relaciones familiares sean buenas es importante.",
    "¿Te cuesta mostrar afecto a tu familia?" ]],

  [r'(.*) nin[oa](.*)',
  [  "¿Tenías amigos íntimos durante tu infancia?",
    "¿Cuál es tu recuerdo favorito de tu infancia?",
    "¿Te acuerdas de algún sueño o pesadilla de tu infancia?",
    "¿Algún niño o niña se metía contigo de pequeños?",
    "¿Cómo piensas que tu infancia ha afectado a tus sentimientos hoy?"]],

  [r'(.*)\?',
  [  "¿Por qué me preguntas eso?",
    "Déjame sugerirte que consideres si tú puedes responder esa pregunta",
    "¿Quizás la respuesta la puedas encontrar dentro de ti?",
    "¿Por qué no me lo dices tú?",
    "%1?"]],

  [r'adios',
  [  "Gracias por hablar conmigo.",
    "Bye bye.",
    "Gracias, serán 200€. Qué tengas buen dia!"]],

  [r'(.*)',
  [  "Por favor, continúa.",
     "Por favor, prosigue.",
    "Cambiemos el foco un poco... Cuéntame sobre tu familia.",
    "¿Puedes expandir un poco sobre eso?",
    "Cambiemos el foco un poco... Cuéntame sobre tu infancia.",
    "¿Puedes expandir un poco sobre eso?",
    "¿Por qué dices %1?",
    "Comprendo.",
    "Muy interesante.",
    "Ya veo.",
    "\"%1\".",
    "Entiendo.  ¿Y eso qué te dice?",
    "¿Cómo hace sentirte eso?",
    "¿Cómo te sientes cuando dices eso?"]]
  ]
