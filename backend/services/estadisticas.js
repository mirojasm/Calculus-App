
const fs = require('fs');

// Clase para calcular métricas de comparación
class ScoreComparator {
  constructor() {
    this.humanScores = [];
    this.aiScores = [];
    this.skillIds = [];
  }

  // Cargar datos desde archivos JSON
  loadData(humanJsonPath, aiJsonPath) {
    try {
      const humanData = JSON.parse(fs.readFileSync(humanJsonPath, 'utf8'));
      const aiData = JSON.parse(fs.readFileSync(aiJsonPath, 'utf8'));
      
      // Extraer puntajes para las habilidades comunes
      const commonSkills = Object.keys(humanData).filter(skill => 
        skill in aiData && humanData[skill].H_Int !== undefined && aiData[skill].H_Int !== undefined
      );
      
      this.skillIds = commonSkills;
      this.humanScores = commonSkills.map(skill => humanData[skill].H_Int);
      this.aiScores = commonSkills.map(skill => aiData[skill].H_Int);
      
      console.log(`✅ Datos cargados: ${commonSkills.length} habilidades comunes`);
      console.log(`📊 Habilidades: ${commonSkills.join(', ')}`);
      
      return true;
    } catch (error) {
      console.error('❌ Error cargando datos:', error.message);
      return false;
    }
  }

  // Calcular MSE (Mean Squared Error)
  calculateMSE() {
    if (this.humanScores.length === 0) return null;
    
    const squaredErrors = this.humanScores.map((human, i) => {
      const ai = this.aiScores[i];
      return Math.pow(human - ai, 2);
    });
    
    const mse = squaredErrors.reduce((sum, error) => sum + error, 0) / squaredErrors.length;
    return mse;
  }

  // Calcular Coeficiente de Correlación de Pearson
  calculatePearsonR() {
    if (this.humanScores.length === 0) return null;
    
    const n = this.humanScores.length;
    const humanMean = this.humanScores.reduce((sum, score) => sum + score, 0) / n;
    const aiMean = this.aiScores.reduce((sum, score) => sum + score, 0) / n;
    
    let numerator = 0;
    let humanSumSquares = 0;
    let aiSumSquares = 0;
    
    for (let i = 0; i < n; i++) {
      const humanDiff = this.humanScores[i] - humanMean;
      const aiDiff = this.aiScores[i] - aiMean;
      
      numerator += humanDiff * aiDiff;
      humanSumSquares += humanDiff * humanDiff;
      aiSumSquares += aiDiff * aiDiff;
    }
    
    const denominator = Math.sqrt(humanSumSquares * aiSumSquares);
    
    if (denominator === 0) return 0;
    
    return numerator / denominator;
  }

  // Calcular ICC (Intraclass Correlation Coefficient)
  calculateICC() {
    if (this.humanScores.length === 0) return null;
    
    const n = this.humanScores.length;
    const k = 2; // Dos evaluadores (humano y IA)
    
    // Calcular medias
    const rowMeans = this.humanScores.map((human, i) => (human + this.aiScores[i]) / 2);
    const grandMean = rowMeans.reduce((sum, mean) => sum + mean, 0) / n;
    
    // Calcular componentes de varianza
    let MSR = 0; // Mean Square for Rows
    let MSE = 0; // Mean Square Error
    let MSC = 0; // Mean Square for Columns
    
    // MSR (Between-subject variance)
    for (let i = 0; i < n; i++) {
      MSR += Math.pow(rowMeans[i] - grandMean, 2);
    }
    MSR = (MSR * k) / (n - 1);
    
    // MSE (Within-subject variance)
    for (let i = 0; i < n; i++) {
      MSE += Math.pow(this.humanScores[i] - rowMeans[i], 2);
      MSE += Math.pow(this.aiScores[i] - rowMeans[i], 2);
    }
    MSE = MSE / (n * (k - 1));
    
    // MSC (Between-rater variance)
    const humanMean = this.humanScores.reduce((sum, score) => sum + score, 0) / n;
    const aiMean = this.aiScores.reduce((sum, score) => sum + score, 0) / n;
    MSC = n * (Math.pow(humanMean - grandMean, 2) + Math.pow(aiMean - grandMean, 2)) / (k - 1);
    
    // Calcular ICC(2,1) - Two-way random effects, single measures, absolute agreement
    const icc = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n);
    
    return icc;
  }

  // Generar reporte completo
  generateReport() {
    if (this.humanScores.length === 0) {
      console.log('❌ No hay datos para generar reporte');
      return null;
    }

    const mse = this.calculateMSE();
    const pearsonR = this.calculatePearsonR();
    const icc = this.calculateICC();
    
    const report = {
      summary: {
        totalSkills: this.skillIds.length,
        humanScores: this.humanScores,
        aiScores: this.aiScores,
        skillIds: this.skillIds
      },
      metrics: {
        MSE: {
          value: mse,
          interpretation: mse < 100 ? 'Buena precisión' : mse < 400 ? 'Precisión moderada' : 'Baja precisión'
        },
        PearsonR: {
          value: pearsonR,
          interpretation: Math.abs(pearsonR) > 0.7 ? 'Correlación fuerte' : 
                         Math.abs(pearsonR) > 0.3 ? 'Correlación moderada' : 'Correlación débil'
        },
        ICC: {
          value: icc,
          interpretation: icc > 0.75 ? 'Excelente concordancia' : 
                         icc > 0.6 ? 'Buena concordancia' : 
                         icc > 0.4 ? 'Concordancia moderada' : 'Pobre concordancia'
        }
      },
      detailedComparison: this.skillIds.map((skill, i) => ({
        skill: skill,
        human: this.humanScores[i],
        ai: this.aiScores[i],
        difference: Math.abs(this.humanScores[i] - this.aiScores[i]),
        percentageError: Math.abs((this.humanScores[i] - this.aiScores[i]) / this.humanScores[i]) * 100
      }))
    };

    return report;
  }

  // Mostrar reporte formateado
  printReport() {
    const report = this.generateReport();
    if (!report) return;

    console.log('\n🔍 REPORTE DE COMPARACIÓN: HUMANO vs IA');
    console.log('=' .repeat(50));
    
    console.log(`\n📊 RESUMEN:`);
    console.log(`   • Total de habilidades: ${report.summary.totalSkills}`);
    console.log(`   • Habilidades evaluadas: ${report.summary.skillIds.join(', ')}`);
    
    console.log(`\n📈 MÉTRICAS ESTADÍSTICAS:`);
    console.log(`   • MSE (Error Cuadrático Medio): ${report.metrics.MSE.value.toFixed(4)}`);
    console.log(`     → ${report.metrics.MSE.interpretation}`);
    console.log(`   • Pearson R (Correlación): ${report.metrics.PearsonR.value.toFixed(4)}`);
    console.log(`     → ${report.metrics.PearsonR.interpretation}`);
    console.log(`   • ICC (Concordancia): ${report.metrics.ICC.value.toFixed(4)}`);
    console.log(`     → ${report.metrics.ICC.interpretation}`);
    
    console.log(`\n🔍 COMPARACIÓN DETALLADA:`);
    console.log('Habilidad\tHumano\tIA\tDiferencia\t% Error');
    console.log('-' .repeat(60));
    
    report.detailedComparison.forEach(item => {
      console.log(`${item.skill}\t\t${item.human.toFixed(2)}\t${item.ai.toFixed(2)}\t${item.difference.toFixed(2)}\t\t${item.percentageError.toFixed(1)}%`);
    });

    // Guardar reporte en JSON
    fs.writeFileSync('comparison_report.json', JSON.stringify(report, null, 2));
    console.log('\n💾 Reporte guardado en: comparison_report.json');
  }
}

// Ejemplo de uso
const comparator = new ScoreComparator();

// Cargar datos desde archivos JSON
// Reemplaza con las rutas reales de tus archivos
const humanJsonPath = '../data/human_scores.json';
const aiJsonPath = '../data/ai_scores.json';

if (comparator.loadData(humanJsonPath, aiJsonPath)) {
  comparator.printReport();
} else {
  console.log('❌ No se pudieron cargar los datos');
}

// Función auxiliar para crear archivos de ejemplo
function createExampleFiles() {
  const exampleHumanScores = {
    "A1": {"H_Freq": 15.25, "H_Qual_Norm": 66.67, "H_Int": 40.96},
    "A2": {"H_Freq": 8.47, "H_Qual_Norm": 55.56, "H_Int": 32.01},
    "B1": {"H_Freq": 12.71, "H_Qual_Norm": 72.22, "H_Int": 42.47},
    "C2": {"H_Freq": 5.00, "H_Qual_Norm": 80.00, "H_Int": 50.00}
  };

  const exampleAiScores = {
    "A1": {"H_Freq": 14.80, "H_Qual_Norm": 70.00, "H_Int": 42.40},
    "A2": {"H_Freq": 9.00, "H_Qual_Norm": 58.33, "H_Int": 33.67},
    "B1": {"H_Freq": 11.95, "H_Qual_Norm": 75.00, "H_Int": 43.48},
    "C2": {"H_Freq": 4.50, "H_Qual_Norm": 83.33, "H_Int": 51.80}
  };

  fs.writeFileSync('example_human_scores.json', JSON.stringify(exampleHumanScores, null, 2));
  fs.writeFileSync('example_ai_scores.json', JSON.stringify(exampleAiScores, null, 2));
  
  console.log('✅ Archivos de ejemplo creados: example_human_scores.json, example_ai_scores.json');
}