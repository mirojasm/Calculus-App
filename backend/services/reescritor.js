// archivo: services/reescritor.js

const OpenAI = require("openai");

class Reescritor {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    this.rewritePrompt = `
Tu tarea es reescribir respuestas para que suenen como las de un compañero de estudio, no como una inteligencia artificial.

Instrucciones importantes:
-Habla como Chileno.
-Incluye las formulas.
-Elimina las palabras o frases que suenen como un robot o innecesarias.
- Mantén el contenido educativo original, pero cambia el tono para que sea de un amigo y colaborativo.
- Usa expresiones naturales, como: "yo creo que tenemos...", "quizás podríamos...", "a mí me sirvió hacer esto...".
- Si el estudiante habla de manera informal, adapta el vocabulario para que también suene relajado.
- No repitas la respuesta original, devuélvela modificada.
- No expliques lo que estás haciendo. Solo devuelve la versión reescrita.
`;
  }

  async reescribir(botResponse) {
    try {
      const messages = [
        {
          role: "system",
          content: this.rewritePrompt,
        },
        {
            role: "user",
            content: `Aquí está la respuesta original:\n\n${botResponse}\n\nPor favor, reescríbela para que suene más como un compañero de estudio.`,
        }
        
        
      ];

      const response = await this.openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages,
        temperature: 0.7,
        max_tokens: 400,
      });

      return response.choices[0].message.content;
    } catch (error) {
      console.error("Error en el reescritor de respuestas:", error);
      return botResponse; // Fallback a la original si hay error
    }
  }
}

module.exports = new Reescritor();
