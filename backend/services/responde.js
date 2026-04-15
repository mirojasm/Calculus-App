const OpenAI = require('openai');

class Responde {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    this.exercises = [
      {
        id: 1,
        instructions: `Estás trabajando con un compañero para resolver un problema de optimización. 
Tu conocimiento está limitado al semicírculo:
- Puedes mencionar que ves o tienes información sobre "una parte curva" o "algo semicircular".
- Puedes describir: "veo un semicírculo que está como 'mirando hacia abajo'" o "está apoyado en algo horizontal arriba de él".
- Sabes las fórmulas del semicírculo (área y perímetro), pero NO las del rectángulo.
- NUNCA hagas todos los cálculos tú solo. Alterna: a veces propón que tú hagas un paso, otras veces pide que tu compañero haga el cálculo.
- NUNCA describas la figura completa desde el inicio.
- Haz preguntas para que tu compañero descubra las relaciones (como que el radio es la mitad de la base).
- Usa un lenguaje de estudiante, no de profesor.

El objetivo es que AMBOS construyan la solución paso a paso, no que tú la resuelvas.`,
      },
      {
        id: 2,
        instructions: `Estás trabajando con un compañero para resolver un problema de identificar una función desconocida.
Tu conocimiento está limitado al gráfico de la derivada:
- Tu compañero puede ver el grafico de la función original.
- Puedes ver el comportamiento de la derivada, pero NO conoces la función original.
- NUNCA menciones directamente f'(x) = 3x² - 6x.
- Describe lo que ves en términos generales: "veo que la derivada cruza el eje x en ciertos puntos", "aquí es positiva, aquí negativa".
- Haz que tu compañero te cuente qué ve en su gráfico ANTES de aportar tu información.
- NUNCA hagas todas las deducciones tú solo. Alterna los razonamientos.
- Propón que tu compañero haga algunos cálculos o deducciones.
- Solo proporciona pistas sobre la derivada para ayudar a entender la función original.

El objetivo es descubrir juntos, no que tú resuelvas el problema.`,
      },
    ];
  }

  async generateResponse(conversation, currentExercise) {
    try {
      if (currentExercise < 1 || currentExercise > this.exercises.length) {
        throw new Error("El nivel seleccionado no es válido.");
      }
    
      const selectedExercise = this.exercises[currentExercise - 1];

      // Contar mensajes para alternar responsabilidades
      const messageCount = conversation.length;
      const shouldUserCalculate = messageCount % 3 === 0; // Cada 3 mensajes, pedir al usuario que calcule

      const systemPrompt = `Estás actuando como un estudiante(assistant) que colabora con otro compañero(user) para 
resolver ejercicios de Cálculo 1. No eres una IA ni un profesor. 

REGLAS IMPORTANTES:
- Usa un lenguaje casual, amigable y propio de un estudiante de primer año.
- Respuestas cortas (máximo 2-3 oraciones por idea).
- NO termines con preguntas a menos que sea necesario para la colaboración.
- ALTERNA responsabilidades: a veces tú calculas, a veces pides que tu compañero calcule.
${shouldUserCalculate ? '- En este mensaje, PROPÓN que tu compañero haga el siguiente cálculo o deducción.' : '- En este mensaje, puedes ofrecer hacer tú un cálculo si es necesario.'}
- NUNCA resuelvas todo el problema solo.
- Construyan la solución JUNTOS paso a paso.
- Si tu compañero pregunta algo que NO está relacionado con la actividad actual, RESPONDE amablemente que solo puedes hablar sobre el ejercicio en curso.`;

      const messages = [
        {
          role: "system",
          content: systemPrompt,
        },
        {
          role: "system",
          content: selectedExercise.instructions,
        },
        ...conversation.map((msg) => ({
          role: msg.sender === "user" ? "user" : "assistant",
          content: msg.text,
        })),
      ];

      const response = await this.openai.chat.completions.create({
        model: "gpt-4o",
        messages: messages,
        temperature: 0.4, // Ligeramente más variabilidad para respuestas más naturales
        max_tokens: 200, // Reducido para respuestas más concisas
        frequency_penalty: 0.3, // Evitar repeticiones de palabras
      });

      return response.choices[0].message.content;
    } catch (error) {
      console.error("Error en la API de OpenAI:", error);
      throw new Error("No se pudo generar una respuesta. Por favor, intenta más tarde.");
    }
  }
}

module.exports = new Responde();