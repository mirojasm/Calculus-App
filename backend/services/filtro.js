// archivo: services/reescritor.js

const OpenAI = require("openai");

class Filtro {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    this.filtroPrompt = `
Eres un filtro de contexto para una aplicación donde un estudiante interactúa con un chatbot que simula ser su compañero de curso en Cálculo 1.

Tu única tarea es determinar si el mensaje del estudiante es apropiado para este contexto académico. NO debes permitir temas personales, irrelevantes.

**REGLAS CLARAS:**
- No expliques, no justifiques, no converses.

**Responde únicamente con:**
- \`true\` → si el mensaje es apropiado.
- (nada) → si el mensaje es inapropiado.

**Mensajes válidos:**
- “No entiendo cómo aplicar la regla de la cadena aquí.”
- “¿Puedes ayudarme con este límite?”
- “Creo que derivé mal, mira: f'(x) = 2x.”

**Mensajes NO válidos:**
- “¿Quién es Cristiano Ronaldo?”
- “Cuéntame algo gracioso.”
- “¿Cuál es tu color favorito?”
- “¿Qué opinas del reguetón?”
- “Háblame de la Segunda Guerra Mundial.”
- “¿Qué harías si tuvieras un millón de dólares?”
- “¿Cuál es tu comida favorita?”
- “¿Qué piensas de la política actual?”
- “¿Cómo te va en la vida?”
- “¿Qué opinas de la película X?”
- “¿Cuál es tu hobby?”
- “¿Qué harías si fueras presidente?”
- “¿Cuál es tu serie favorita?”
- “¿Qué piensas de la música pop?”
- “¿Qué opinas de la inteligencia artificial?”
- “¿Cuál es tu libro favorito?”
- “¿Qué harías en un apocalipsis zombie?“  
-“¿Quien es x persona?”
- “¿Qué opinas de x juego?”
- “¿Qué opinas de la moda actual?”
- “¿Cuál es tu lugar favorito para viajar?”
- “¿Qué harías si tuvieras un día libre?”
- “¿Qué opinas de la comida rápida?”
- “¿Cuál es tu animal favorito?”
- “¿Qué opinas de la educación en línea?”
- “¿Qué harías si fueras invisible por un día?”

Evalúa ahora.
`;

  }
  async moderar(mensajeUsuario) {
  try {
    const result = await this.openai.moderations.create({
      input: mensajeUsuario
    });
    return result.results[0]; // contiene: flagged (boolean), categories, etc.
  } catch (error) {
    console.error("Error en la moderación:", error);
    return { flagged: false }; // fallback en caso de error
  }
}

  async filtrar(mensajeUsuario) {
  try {
    // Paso 1: Moderación general
    const moderacion = await this.moderar(mensajeUsuario);
    if (moderacion.flagged) {
      console.warn("Mensaje bloqueado por moderación de OpenAI.");
      return ""; // mensaje inapropiado
    }else{
        return "true"
    }

  } catch (error) {
    console.error("Error en el filtro:", error);
    return ""; // Fallback seguro
  }
}

}

module.exports = new Filtro();