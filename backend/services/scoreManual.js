const fs = require('fs');

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
    this.logData = ""; // Variable para almacenar los logs
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

      const logMessage = [
        `📌 ${skillId} - ${CPS_RUBRIC_DETAILED[skillId].name}`,
        `  → Frecuencia: ${data.freq}`,
        `  → Total contribuciones: ${this.totalContributions}`,
        `  → hFreq (%): ${hFreq.toFixed(2)}`,
        `  → hQual promedio: ${hQual.toFixed(2)}`,
        `  → hQualNorm (%): ${hQualNorm.toFixed(2)}`,
        `  → Índice final (H_Int): ${hInt.toFixed(2)}`,
        '----------------------'
      ].join('\n');

      console.log(logMessage); // Imprimir en consola
      this.logData += logMessage + '\n'; // Guardar en la variable logData

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

// Leer el archivo de codificaciones
const codificaciones = JSON.parse(fs.readFileSync('../conversaciones/vicente_liberona.json', 'utf8'));

// Crear una instancia de StudentScorecard
const scorecard = new StudentScorecard("vicente_liberona");

// Agregar contribuciones
codificaciones.forEach(entry => {
  if (entry.skill_id && typeof entry.quality_score === 'number') {
    scorecard.addContribution(entry.skill_id, entry.quality_score);
  }
});

// Calcular los puntajes finales
const finalScores = scorecard.calculateFinalScores();
console.log("✅ Puntajes manuales:", finalScores);

// Agregar el JSON de puntajes manuales al logData
scorecard.logData += '\n✅ Puntajes manuales:\n';
scorecard.logData += JSON.stringify(finalScores, null, 2) + '\n';

// Escribir el contenido de los logs en un archivo .txt
const logFilePath = `../conversaciones/${scorecard.studentName}.txt`;
fs.writeFileSync(logFilePath, scorecard.logData, 'utf8');
console.log(`📄 Logs guardados en: ${logFilePath}`);
