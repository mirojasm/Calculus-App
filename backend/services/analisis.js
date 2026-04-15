const OpenAI = require('openai');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ------------------- Rubrica detallada -------------------
const CPS_RUBRIC_DETAILED = {
    "A1": {"name": "Discovering perspectives and abilities", "indicators": ["IND-A1.1: Asks teammates directly what they know about the problem or its objectives.", "IND-A1.2: Asks teammates for their point of view or how they interpret the problem.", "IND-A1.3: Proactively states their own initial knowledge or perspective on the problem."]},
    "A2": {"name": "Discovering the type of collaborative interaction", "indicators": ["IND-A2.1: Asks about or suggests specific actions needed to start solving the problem.", "IND-A2.2: Identifies and communicates the problem's constraints or rules that affect the group's actions."]},
    "A3": {"name": "Understanding roles to solve the problem", "indicators": ["IND-A3.1: Suggests or asks about the roles needed to tackle the problem (e.g., 'Who can be the researcher?').", "IND-A3.2: Asks about teammates' strengths or mentions how their own skills can contribute."]},
    "B1": {"name": "Building a shared representation and negotiating meaning", "indicators": ["IND-B1.1: Paraphrases or summarizes a teammate's idea to verify mutual understanding.", "IND-B1.2: Explicitly states agreement or disagreement on the problem's interpretation to negotiate a common view.", "IND-B1.3: Provides summaries of the group's understanding ('So, what we all agree on is...')."]},
    "B2": {"name": "Identifying and describing tasks to be completed", "indicators": ["IND-B2.1: Proposes specific and measurable tasks or goals for the group.", "IND-B2.2: Actively participates in the discussion to define and refine the team's task list."]},
    "B3": {"name": "Describing roles and team organization", "indicators": ["IND-B3.1: Proposes an explicit distribution of tasks or assignment of roles.", "IND-B3.2: Negotiates their own role or others' based on the defined tasks and team skills."]},
    "C1": {"name": "Communicating with team members about actions to be performed", "indicators": ["IND-C1.1: Proposes a plan or a sequence of steps for the solution.", "IND-C1.2: Integrates suggestions from others to build or improve the group's action plan.", "IND-C1.3: Actively asks for the group's agreement on the proposed plan ('Are we all on board?')."]},
    "C2": {"name": "Enacting plans", "indicators": ["IND-C2.1: Performs a concrete action that is part of the agreed plan (e.g., manipulates a tool, writes a shared response).", "IND-C2.2: Verbally communicates that they have completed their part of a task ('Done with my research on X')."]},
    "C3": {"name": "Following rules of engagement", "indicators": ["IND-C3.1: Constructively encourages or reminds a teammate to perform their assigned task.", "IND-C3.2: Keeps communication channels open, asks follow-up questions, and ensures important information is shared."]},
    "D1": {"name": "Monitoring and repairing the shared understanding", "indicators": ["IND-D1.1: Detects and points out a possible confusion or misunderstanding in the group ('I think we might be talking about two different things...').", "IND-D1.2: Offers or requests a clarification to repair an identified misunderstanding."]},
    "D2": {"name": "Monitoring results of actions and evaluating success", "indicators": ["IND-D2.1: Compares the result of an action against the group's goals ('This didn't work as expected because...').", "IND-D2.2: Offers constructive feedback to a teammate on the effectiveness of their action.", "IND-D2.3: Asks the group if the current plan is working or if progress is being made."]},
    "D3": {"name": "Monitoring, providing feedback and adapting the team organization", "indicators": ["IND-D3.1: Points out that the current team organization or role distribution is not working.", "IND-D3.2: Proposes a change in the team's strategy, roles, or processes to overcome an obstacle."]}
};

// ------------------- Clase Scorecard (sin cambios) -------------------
class StudentScorecard {
  constructor(studentName) {
    this.studentName = studentName;
    this.totalContributions = 0;
    this.skills = {};

    const skillKeys = Object.keys(CPS_RUBRIC_DETAILED);
    skillKeys.forEach(key => {
      this.skills[key] = { freq: 0, quality_scores: [] };
    });

    this.finalScores = {};
  }

  addContribution(skillId, qualityScore) {
    if (this.skills[skillId]) {
      this.skills[skillId].freq += 1;
      this.skills[skillId].quality_scores.push(qualityScore);
      this.totalContributions += 1;
    }
  }

  calculateFinalScores() {
    if (this.totalContributions === 0) return null;

    for (const skillId in this.skills) {
      const data = this.skills[skillId];
      let hFreq, hQual;

      if (skillId === "C2") {
        hFreq = data.freq;
      } else {
        hFreq = (data.freq / this.totalContributions) * 100;
      }

      hQual = data.quality_scores.length
        ? data.quality_scores.reduce((a, b) => a + b, 0) / data.quality_scores.length
        : 0;

      const hQualNorm = (hQual / 3) * 100;
      const wF = skillId === "C2" ? 0.4 : 0.5;
      const wC = skillId === "C2" ? 0.6 : 0.5;

      const hFreqNorm = skillId === "C2"
        ? (hFreq / 10) * 100
        : hFreq;

      const hInt = (wF * hFreqNorm) + (wC * hQualNorm);

      this.finalScores[skillId] = {
        H_Freq: Number(hFreq.toFixed(2)),
        H_Qual_Norm: Number(hQualNorm.toFixed(2)),
        H_Int: Number(hInt.toFixed(2))
      };
    }

    return this.finalScores;
  }

  getSummaryForFeedback() {
    return JSON.stringify(this.finalScores, null, 2);
  }
}

function convertChatLogToFormat(chatLog, userId) {
  console.log(`🗂️ Log original: ${chatLog.length} entradas.`);
  const sortedLog = chatLog.sort((a, b) => new Date(a.creado_en) - new Date(b.creado_en));
  const formattedLog = [];

  sortedLog.forEach(entry => {
    if (entry.usuario_id === userId && entry.mensaje_usuario?.trim()) {
      formattedLog.push({
        author: `Usuario_${userId}`,
        text: entry.mensaje_usuario,
        exercise: entry.currentexercise
      });
      if (entry.respuesta_chatbot?.trim()) {
        formattedLog.push({
          author: "Chatbot",
          text: entry.respuesta_chatbot,
          exercise: entry.currentexercise
        });
      }
    }
  });

  console.log(`📊 Mensajes válidos del usuario: ${formattedLog.filter(m => m.author === `Usuario_${userId}`).length}`);
  return formattedLog;
}

async function generateFeedbackForChatbotLog(originalChatLog, targetUserId) {
  const chatLog = convertChatLogToFormat(originalChatLog, targetUserId);
  const targetStudent = `Usuario_${targetUserId}`;

  if (!chatLog.some(m => m.author === targetStudent)) {
    return "No hay mensajes válidos del usuario para analizar.";
  }

  // Formatear la rúbrica completa para el contexto
  const formattedRubric = Object.entries(CPS_RUBRIC_DETAILED)
    .map(([id, obj]) => `**${id}: ${obj.name}**\n` + obj.indicators.map(i => `- ${i}`).join("\n"))
    .join("\n\n");

  const systemMessage = { role: "system", content: `You are a collaborative problem-solving evaluator.\n\n${formattedRubric}` };

  const scorecard = new StudentScorecard(targetStudent);
  const context = [];

  // Análisis de cada mensaje (sin cambios)
  for (const message of chatLog) {
    if (message.author === targetStudent) {
      console.log(`🔍 Analizando mensaje: "${message.text.slice(0, 60)}..."`);

      const userPrompt = `
Context (últimos 5 mensajes):
${JSON.stringify(context.slice(-5), null, 2)}

Mensaje actual de ${targetStudent}:
"${message.text}"

Exercise: ${message.exercise}

Devuelve solo JSON así:
{"skill_id":"C1","indicator_id":"IND-C1.3","quality_score":3,"reasoning":"..."}
Si no hay habilidad CPS, usa: {"skill_id":null,"indicator_id":null,"quality_score":0,"reasoning":"No clear CPS skill demonstrated"}
      `.trim();

      let rawResponse;
      try {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [systemMessage, { role: "user", content: userPrompt }],
          temperature: 0
        });
        rawResponse = response.choices[0].message.content.trim();
      } catch (err) {
        console.error("❌ Error en la API de OpenAI:", err.message);
        continue;
      }

      let parsed;
      try {
        const cleaned = rawResponse.replace(/```json|```/g, "").trim();
        parsed = JSON.parse(cleaned);
      } catch (err) {
        console.error("❌ Error al parsear JSON:", err.message);
        console.error("🔍 Respuesta cruda:", rawResponse);
        continue;
      }

      if (parsed.skill_id && typeof parsed.quality_score === "number") {
        scorecard.addContribution(parsed.skill_id, parsed.quality_score);
        console.log(`✅ Registrado: ${parsed.skill_id} con score ${parsed.quality_score}`);
      } else {
        console.warn("⚠️ Respuesta sin skill_id/quality_score válido:", parsed);
      }

      context.push(message);
    }
  }

  const finalScores = scorecard.calculateFinalScores();
  if (!finalScores) return "No se pudo generar feedback: ninguna contribución válida.";


const feedbackPrompt = `
Genera un reporte de retroalimentación detallado y constructivo para un estudiante .
Asume el rol de un coach educativo experto. El tono debe ser de apoyo y alentador.

Usa los siguientes datos para escribir tu reporte.

### DATOS DE ENTRADA 1: Scorecard Final del Estudiante
\`\`\`json
${scorecard.getSummaryForFeedback()}
\`\`\`

### DATOS DE ENTRADA 2: Log Completo de la Conversación
\`\`\`json
${JSON.stringify(chatLog, null, 2)}
\`\`\`

### DATOS DE ENTRADA 3: Referencia de Habilidades CPS
${formattedRubric}

### TAREA: Generar el Reporte de Retroalimentación Completo
El reporte DEBE seguir esta estructura:
* **Parte A: Análisis del Perfil General:** Basado en los puntajes promedio generales, determina el perfil del estudiante (Líder Colaborativo, Participante Entusiasta, Pensador Silencioso, u Observador Pasivo).
* **Parte B: Fortalezas Clave:** Identifica las 2-3 habilidades con los puntajes \`H_Int\` más altos. Para cada fortaleza, describe lo que el estudiante hizo bien y cita un mensaje específico del chat como ejemplo positivo.
* **Parte C: Áreas de Crecimiento:** Identifica las 2-3 habilidades con los puntajes \`H_Int\` más bajos. Para cada área, explica qué faltó y refiere a un momento en el chat donde esta habilidad podría haberse usado.
* **Parte D: Recomendaciones Accionables:** Proporciona 2-3 consejos claros, orientados al futuro, directamente vinculados a mejorar las "Áreas de Crecimiento".

Contexto: Esta es una conversación entre un estudiante y un chatbot mientras resuelve problemas de cálculo.

Basado en todos los datos proporcionados, escribe ahora el reporte de retroalimentación completo.
  `.trim();

  let feedback;
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: "Eres un coach educativo experto que genera retroalimentación detallada y constructiva sobre habilidades de resolución colaborativa de problemas. Siempre respondes en español." },
        { role: "user", content: feedbackPrompt }
      ],
      temperature: 0.7,
      max_tokens: 1500 
    });
    feedback = response.choices[0].message.content.trim();
  } catch (err) {
    console.error("❌ Error generando feedback final:", err.message);
    feedback = "Hubo un error al generar el feedback final.";
  }

  return { feedback, scores: finalScores, totalMessages: scorecard.totalContributions, studentName: targetStudent };
}

module.exports.analyzeChatbotLog = async function (chatLog, userId) {
  console.log(`🚀 Iniciando análisis para usuario ${userId}`);
  try {
    const result = await generateFeedbackForChatbotLog(chatLog, userId);
    console.log("✅ Análisis completado:", JSON.stringify(result.scores, null, 2));
    console.log("📝 Feedback generado:", result.feedback);
    return result;
  } catch (err) {
    console.error("❌ Error en el análisis:", err);
    return null;
  }
};
