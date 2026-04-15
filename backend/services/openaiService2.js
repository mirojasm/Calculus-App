const OpenAI = require('openai');

class OpenAIService {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

  }

  async generateResponse(conversaciones) {
  try {
    // Convertimos el arreglo en texto plano con formato de diálogo
    const conversacionFormateada = conversaciones.map(item => {
      return `Usuario: ${item.mensaje_usuario}\nChatbot: ${item.respuesta_chatbot}`;
    }).join('\n\n');

    const messages = [
      {
        role: "system",
        content: `Eres un evaluador experto en colaboración entre humanos y chatbots. Analiza la calidad de la colaboración en la siguiente conversación en el contexto de una aplicación de aprendizaje de Cálculo 1. Evalúa aspectos como: 
- iniciativa del usuario,
- apoyo del chatbot,
- resolución conjunta,
- toma de decisiones,
- comunicación efectiva.

Proporciona un análisis detallado, incluyendo fortalezas, debilidades y una conclusión general.`,
      },
      {
        role: "user",
        content: `Aquí está la conversación:\n\n${conversacionFormateada}\n\nPor favor, realiza el análisis de colaboración.`,
      },
    ];

    const response = await this.openai.chat.completions.create({
      model: "gpt-4",
      messages: messages,
      temperature: 0.5,
      max_tokens: 500,
    });

    return response.choices[0].message.content;
  } catch (error) {
    console.error("Error en la API de OpenAI:", error);
    throw new Error("No se pudo generar una respuesta. Por favor, intenta más tarde.");
  }
}

}

module.exports = new OpenAIService();
