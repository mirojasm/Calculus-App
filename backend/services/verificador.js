const OpenAI = require('openai');

class Verificador {
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
    instructions: `Se dispone del gráfico de una función desconocida, sin contar con su expresión algebraica ni con su derivada.
El objetivo es identificar, con la mayor precisión posible, cuál es dicha función.
Para ello, se trabaja en colaboración: una parte del equipo tiene acceso únicamente al gráfico de la función, mientras que la otra cuenta solo con el gráfico de su derivada.
La tarea consiste en observar detenidamente el gráfico disponible, describiendo aspectos como la forma general de la curva, la ubicación de máximos y mínimos, y los cambios en la concavidad.
A partir de este intercambio de información, se busca reconstruir e identificar conjuntamente la función original.`,
  },
  
];
   this.collaborationPrompt = `
Eres un agente evaluador silencioso. Tu única tarea es revisar una conversación entre un estudiante (user) y un compañero (assistant).

Tu tarea es determinar si llegaron a un acuerdo sobre una solución matemática.

Responde solo con una de las siguientes opciones:
- "ACUERDO_CLARO": Si identificas que llegaron a una solución consensuada.
- "SIN_ACUERDO": Si no hay evidencia clara de que hayan consensuado una respuesta.
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
        content: `Aquí está la conversación entre el estudiante y su compañero:\n\n${conversation.map(msg => `${msg.sender === "user" ? "Estudiante" : "Compañero"}: ${msg.text}`).join('\n')}\n\nPor favor, determina si llegaron a un acuerdo claro sobre una solución matemática.`,
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

module.exports = new Verificador();