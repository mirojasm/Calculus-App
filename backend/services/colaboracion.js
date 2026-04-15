const OpenAI = require('openai');

class AgenteColaborativo {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

   this.collaborationPrompt = `
Eres un agente evaluador silencioso. Tu única tarea es revisar una conversación entre un estudiante (user) y un compañero (assistant).

Tu función: detectar si el estudiante ha avanzado a una nueva fase de colaboración, según el siguiente orden:

1. El usuario expresa una duda sobre el problema. Marca: [PHASE_REACHED0]  
2. El usuario propone un razonamiento propio (aunque sea incorrecto). Marca: [PHASE_REACHED1]  
3. El usuario interactúa con la respuesta del compañero. Marca: [PHASE_REACHED2]  
4. El usuario y el compañero llegan juntos a una conclusión razonada. Marca: [PHASE_REACHED3]  
   - Si se completan todas las fases: [COLLAB_DONE]

Reglas estrictas:
- ❌ No escribas explicaciones.
- ❌ No digas “el usuario ha alcanzado...”.
- ❌ No repitas marcas ya alcanzadas (pasadas por parámetro).
- ❌ No simules ser asistente o compañero.
- ✅ Solo responde con las nuevas marcas, una por línea.
- ✅ Si no hay nuevas marcas, responde con una cadena vacía.

Ejemplo válido:
[PHASE_REACHED1]  
[PHASE_REACHED2]

Ejemplo inválido:
“El usuario ha alcanzado la fase 2.” ❌
“Has llegado a una solución.” ❌
`;
  }

  async generateResponse(conversation,fase) {
    try {
      
      const historyText = conversation
  .map((msg) => `${msg.sender === "user" ? "Estudiante" : "Compañero"}: ${msg.text}`)
  .join('\n');

  const messages = [
  { role: "system", content: this.collaborationPrompt },
  {
    role: "user",
    content:
      `Hasta ahora, el usuario ha alcanzado la fase: ${fase}. Solo marca fases superiores si se detecta progreso.\n\n` +
      `Aquí está la conversación:\n${historyText}`,
  },
  ]

      const response = await this.openai.chat.completions.create({
        model: "gpt-4",
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

module.exports = new AgenteColaborativo();