const OpenAI = require('openai');

class Provocador {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

  
this.exercises = [
  {
    id: 1,
    instructions: `Se presenta una figura compuesta por un rectángulo y un semicírculo,
     donde el semicírculo se encuentra apoyado sobre la base inferior del rectángulo, 
    coincidiendo su diámetro con uno de los lados del rectángulo (lado A). 
    El objetivo es determinar los valores de A y B que 
    permiten maximizar el área total de la figura, 
    considerando que el perímetro total es de 20 unidades. 
    Para resolver este problema, cada integrante del equipo posee información parcial: 
    uno puede ver y modificar los valores del rectángulo, mientras que el otro conoce los elementos 
    necesarios para calcular el área y el perímetro del semicírculo. 
    La solución requiere trabajo colaborativo y comunicación efectiva para combinar la información y alcanzar el resultado óptimo.`,
  },
  {
    id: 2,
    instructions: `En este ejercicio a un integrante del equipo se le entregará el gráfico de una función desconocida. Al otro, el gráfico de su derivada. Ninguno contará con la expresión algebraica de la función.
El objetivo es colaborar para identificar, con la mayor precisión posible, cuál podría ser la función original.
Quien tenga el gráfico de la función deberá observarlo con atención y describirlo al compañero: cómo es la forma general de la curva, en qué intervalos sube o baja, dónde se encuentran los máximos y mínimos, y cómo cambia la concavidad.
Quien tenga el gráfico de la derivada utilizará esa información, junto con lo que observa en su propio gráfico, para analizar el comportamiento de la función.
Ambos deben comunicarse, compartir observaciones y reflexionar en conjunto para proponer una reconstrucción razonada de la función original.
`,
  },


  
];
   this.collaborationPrompt = `
Actúas como un compañero más en una conversación grupal de Cálculo 1.

Tu única tarea es intervenir siempre con **una frase breve, casual y colaborativa**, como si fueras un estudiante .

Puede ser UNA de las siguientes:
- Una **pregunta corta y natural** que invite a compartir ideas, coordinar, negociar o construir significado común.
- Un **comentario provocador o empático** que motive a seguir, reflexionar o reorganizarse.

✅ Tu tono debe ser siempre juvenil, amistoso y relajado — como si fueras un estudiante más, no una IA ni un experto.

✅ Siempre debes intervenir. Nunca digas que no tienes algo que decir.

❌ Nunca resuelvas ejercicios, des información matemática, expliques conceptos ni actúes como profesor.

Ejemplos válidos:
- “¿Y si probamos otra forma?”
- “¡Esa parte me confundió, a ti te pasó?”
- “¿Vamos bien o nos enredamos acá?”
- “¿Eso significa que ya casi lo tenemos?”
- “¿Te parece si dividimos lo que falta?”
- “¿Lo estás entendiendo igual que yo?”

No repitas frases exactas de los ejemplos. Varía tu lenguaje de forma natural. Usa muletillas ocasionales como “mmm”, “creo que”, “no sé si les pasa”, “o sea”.

Siempre mantén una actitud colaborativa, cercana y motivadora.


`;
  }

  async generateResponse(conversation,currentExercise) {
    try {

     if (currentExercise < 1 || currentExercise > this.exercises.length) {
        throw new Error("El nivel seleccionado no es válido.");
      }

    const selectedExercise = this.exercises[currentExercise - 1];

  const messages = [
  { role: "system", content: this.collaborationPrompt },
    {
        role: "system",
        content: `Contexto del problema: \n\n${selectedExercise.instructions}`,
    },
    {
        role: "user",
        content: `Aquí está la conversación entre el estudiante y su compañero:\n\n${conversation.map(msg => `${msg.sender === "user" ? "Estudiante" : "Compañero"}: ${msg.text}`).join('\n')}\n\n`,
    },
  ]

      const response = await this.openai.chat.completions.create({
        model: "gpt-4o",
        messages: messages,
        temperature: 0.3,
        max_tokens: 200,
      });

      return response.choices[0].message.content;
    } catch (error) {
      console.error("Error en la API de OpenAI:", error);
      throw new Error("No se pudo generar una respuesta. Por favor, intenta más tarde.");
    }
  }
}

module.exports = new Provocador();