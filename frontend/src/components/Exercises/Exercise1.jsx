import React, { useState, useEffect, useRef } from "react";
import { Button } from "react-bootstrap";
import { MathJax } from "better-react-mathjax";
import "./Exercise.css";
import { accionService } from "../../services/accionService";
import {resultadoService} from "../../services/resultadoService";
import { MathJaxContext } from "better-react-mathjax";

const mathJaxConfig = {
  loader: { load: ["[tex]/ams"] },
  tex: {
    packages: { "[+]": ["ams"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
  },
};
const solucionesLatex = {
  A: "A) \\( A = \\frac{40}{4 + \\pi},\\quad B = \\frac{20}{3 + \\pi} \\)",
  B: "B) \\( A = \\frac{20}{4 + \\pi},\\quad B = \\frac{40}{4 + \\pi} \\)",
  C: "C) \\( A = \\frac{40}{4 + \\pi},\\quad B = \\frac{20}{4 + \\pi} \\)",
};

const Exercise1 = ({ onNext }) => {
  const [collabUnlocked, setCollabUnlocked] = useState(false);
  const [answered, setAnswered] = useState(false);
  const [correct, setCorrect] = useState(null);
  const [showNext, setShowNext] = useState(false);
  const iframeRef = useRef(null);

  const handleAnswer = async (choice) => {
  const isCorrect = choice === "C";
  setAnswered(true);
  setCorrect(isCorrect);
  setShowNext(true);

  const current_exercise = 1;
  const resultado_alumno = solucionesLatex[choice];

  try {
    const resultadoData = await resultadoService(resultado_alumno, current_exercise);
    console.log("Solución enviada:", resultadoData);
  } catch (error) {
    console.error("Error al enviar la solución:", error);
  }
};

  useEffect(() => {
    const handleEvent = () => setCollabUnlocked(true);
    window.addEventListener("collab-unlocked-1", handleEvent);
    return () => {
      window.removeEventListener("collab-unlocked-1", handleEvent);
    };
  }, []);

  useEffect(() => {
    let mouseOverIframe = false;

    const handleMouseEnter = () => {
      mouseOverIframe = true;
      console.log("Mouse entró al iframe");
    };

    const handleMouseLeave = () => {
      mouseOverIframe = false;
      console.log("Mouse salió del iframe");
    };

    const handleWindowBlur = async () => {
      if (mouseOverIframe) {
        const accion = "Click detectado en grafico";
        const current_exercise = 1;
        const accionData = await accionService(accion, current_exercise);
        console.log("acción registrada:", accionData);
      }
    };

    const handleWindowFocus = () => {
      console.log("Ventana recuperó el foco");
    };

    const iframe = iframeRef.current;
    if (iframe) {
      iframe.addEventListener("mouseenter", handleMouseEnter);
      iframe.addEventListener("mouseleave", handleMouseLeave);
    }

    window.addEventListener("blur", handleWindowBlur);
    window.addEventListener("focus", handleWindowFocus);

    return () => {
      if (iframe) {
        iframe.removeEventListener("mouseenter", handleMouseEnter);
        iframe.removeEventListener("mouseleave", handleMouseLeave);
      }
      window.removeEventListener("blur", handleWindowBlur);
      window.removeEventListener("focus", handleWindowFocus);
    };
  }, []);

  const getVariant = (option) => {
    if (!answered) return "outline-primary";
    if (correct && option === "C") return "success";
    if (!correct && option === "C") return "success";
    if (!correct && option !== "C") return "danger";
    return "outline-primary";
  };

  return (
    <div className="exercise-card">
      <div className="exercise-question">
        <h3>Optimización Multizona</h3>
        <p className="mb-3 small-text">
         Junto a tu compañero, deben encontrar los valores de A y B que maximicen el área de una figura compuesta por dos partes: un rectángulo (que tú puedes ver) y una figura adicional (que solo tu compañero puede ver). La figura completa debe tener un perímetro total exacto de 20 unidades.
Desde tu punto de vista, puedes observar un rectángulo con base A y altura B. Para resolver el desafío, será fundamental que ambos compartan la información que tienen y trabajen en conjunto.
        </p>
        <div className="geogebra-container mb-3 px-0 mx-auto">
          <iframe
            ref={iframeRef}
            scrolling="no"
            title="Optimización Multizona"
            src="https://www.geogebra.org/material/iframe/id/gnykmzun/width/800/height/500/border/888888/sfsb/false/smb/false/stb/false/stbh/false/ai/false/asb/false/sri/false/rc/false/ld/false/sdz/false/ctl/false"
            loading="lazy"
            style={{ width: "100%", height: "auto", aspectRatio: "16 / 9", border: "0" }}
          ></iframe>
        </div>
      </div>

      {collabUnlocked ? (
        <MathJaxContext version={3} config={mathJaxConfig}>
        <div className="exercise-options">
          <Button
            variant={getVariant("A")}
            className="option-button"
            onClick={() => handleAnswer("A")}
            disabled={answered}
          >
            <MathJax>
              {"A) \\( A = \\frac{40}{4 + \\pi},\\quad B = \\frac{20}{3 + \\pi} \\)"}
            </MathJax>
          </Button>

          <Button
            variant={getVariant("B")}
            className="option-button"
            onClick={() => handleAnswer("B")}
            disabled={answered}
          >
            <MathJax>
              {"B) \\( A = \\frac{20}{4 + \\pi},\\quad B = \\frac{40}{4 + \\pi} \\)"}
            </MathJax>
          </Button>

          <Button
            variant={getVariant("C")}
            className="option-button"
            onClick={() => handleAnswer("C")}
            disabled={answered}
          >
            <MathJax>
              {"C) \\( A = \\frac{40}{4 + \\pi},\\quad B = \\frac{20}{4 + \\pi} \\)"}
            </MathJax>
          </Button>

         {answered && !correct && (
  <div className="feedback-box incorrect">
    <i className="bi bi-x-circle"></i> 
    <div className="latex-explanation">
      <MathJax>
        {"Para maximizar \\( A_{\\text{total}} \\) respecto a \\( A \\), derivamos y buscamos el valor crítico:"}
      </MathJax>
      <MathJax>
        {"\\[ \\frac{dA_{\\text{total}}}{dA}=10-\\frac{(4+\\pi)2A}{8}=10-\\frac{(4+\\pi)A}{4}=0. \\]"}
      </MathJax>
      <MathJax>
        {"Resolviendo para \\( A \\):"}
      </MathJax>
      <MathJax>
        {"\\[ \\frac{(4+\\pi)A}{4}=10\\quad\\Longrightarrow\\quad A=\\frac{40}{4+\\pi}. \\]"}
      </MathJax>
      <MathJax>
        {"\\[ B=\\frac{20-\\frac{20(2+\\pi)}{4+\\pi}}{2} =\\frac{20\\left(\\frac{(4+\\pi)-(2+\\pi)}{4+\\pi}\\right)}{2} =\\frac{20\\left(\\frac{2}{4+\\pi}\\right)}{2} =\\frac{20}{4+\\pi}. \\]"}
      </MathJax>
    </div>
  </div>
)}

          {answered && correct && (
            <div className="feedback-box correct">
              <i className="bi bi-check-circle"></i> ¡Correcto! Has deducido correctamente las dimensiones óptimas.
            </div>
          )}

          {answered && (
            <Button className="next-button" onClick={onNext}>
              Siguiente ejercicio <i className="bi bi-arrow-right"></i>
            </Button>
          )}
        </div>
        </MathJaxContext> 
      ) : (
       <div className="collaboration-message">
  <div className="collaboration-icon">
    <i className="bi bi-chat-dots"></i>
    <p>
      ¡Colabora con tu compañero para resolver el desafío! <br />
      <strong>Tienen hasta <span style={{ color: "#007bff" }}>30 turnos</span> para llegar juntos a la respuesta final.</strong>
    </p>
    <p className="hint">
      <span style={{ color: "#dc3545", fontWeight: "bold" }}>
        Usa el botón <strong>“Llegamos a un acuerdo”</strong> <u>solo si ambos ya están de acuerdo y quieren responder antes de los 30 turnos</u>.
      </span>
    </p>
    <p className="hint">
      Apóyense usando el gráfico interactivo para visualizar ideas y validar sus propuestas.
    </p>
    <p className="hint" style={{ marginBottom: 0 }}>
      Consejos para colaborar bien:
    </p>
  </div>
  <ol className="collaboration-steps">
    <li>🤝 <strong>Escuchen y comprendan:</strong> Aclaren dudas y asegúrense de que ambos entienden el problema.</li>
    <li>🧠 <strong>Propongan y discutan ideas:</strong> Compartan puntos de vista, usen el gráfico y argumenten sus propuestas.</li>
    <li>✅ <strong>Decidan juntos:</strong> Lleguen a una solución en conjunto y asegúrense de que ambos estén de acuerdo antes de responder.</li>
  </ol>
</div>
      )}
    </div>
  );
};

export default Exercise1;



