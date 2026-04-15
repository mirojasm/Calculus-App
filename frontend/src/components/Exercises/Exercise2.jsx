import React, { useState, useEffect, useRef } from "react";
import { Button } from "react-bootstrap";
import { MathJax } from "better-react-mathjax";
import "./Exercise.css";
import { useNavigate } from "react-router-dom";
import { accionService } from "../../services/accionService";
import { resultadoService } from "../../services/resultadoService";
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
  A: "A) \\( f(x) = x^3 - 3x^2 + 2 \\)",
  B: "B) \\( f(x) = \\frac{1}{x^2 + 1} \\)",
  C: "C) \\( f(x) = 2^x - x \\)",
};

const Exercise2 = ({ onNext }) => {
  const [collabUnlocked, setCollabUnlocked] = useState(false);
  const [answered, setAnswered] = useState(false);
  const [correct, setCorrect] = useState(null);
  const [showNext, setShowNext] = useState(false);
  const iframeRef = useRef(null);
  const navigate = useNavigate();

  const handleAnswer = async (choice) => {
    const isCorrect = choice === "A";
    setAnswered(true);
    setCorrect(isCorrect);
    setShowNext(true);

    const current_exercise = 2;
    const resultado_alumno = solucionesLatex[choice];

    try {
      const resultadoData = await resultadoService(
        resultado_alumno,
        current_exercise
      );
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
    if (correct && option === "A") return "success";
    if (!correct && option === "A") return "success";
    if (!correct && option !== "A") return "danger";
    return "outline-primary";
  };

  return (
    <div className="exercise-card">
      <div className="exercise-question">
        <h3>Función Desconocida</h3>
        <p className="mb-3 small-text">
          Junto con tu compañero, su desafío es descubrir cuál es la función que
          corresponde al gráfico que estás viendo. Tú solo tienes acceso al
          gráfico de la función original. Tu compañero observa un gráfico
          diferente, pero relacionado, que es clave para resolver el problema.
          Deberán comunicarse para descubrir la conexión entre ambos gráficos e
          identificar la función. Abajo puedes ver el gráfico de la función
          original y también puedes usar el botón para graficar una función que
          quieras probar.
        </p>
        <div className="geogebra-container exercise2 mb-3 px-0 mx-auto">
          <iframe
            ref={iframeRef}
            scrolling="no"
            title="Funcion Desconocida"
            src="https://www.geogebra.org/material/iframe/id/mz2z7twc/width/800/height/500/border/888888/sfsb/false/smb/false/stb/false/stbh/false/ai/false/asb/false/sri/false/rc/false/ld/false/sdz/false/ctl/false"
            loading="lazy"
            style={{
              width: "100%",
              height: "auto",
              aspectRatio: "16 / 9",
              border: "0",
            }}
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
              <MathJax>{"A) \\( f(x) = x^3 - 3x^2 + 2 \\)"}</MathJax>
            </Button>

            <Button
              variant={getVariant("B")}
              className="option-button"
              onClick={() => handleAnswer("B")}
              disabled={answered}
            >
              <MathJax>{"B) \\( f(x) = \\frac{1}{x^2 + 1} \\)"}</MathJax>
            </Button>

            <Button
              variant={getVariant("C")}
              className="option-button"
              onClick={() => handleAnswer("C")}
              disabled={answered}
            >
              <MathJax>{"C) \\( f(x) = 2^x - x \\)"}</MathJax>
            </Button>

            {answered && !correct && (
              <div className="feedback-box incorrect">
                <i className="bi bi-x-circle"></i>
                <div className="latex-explanation">
                  <MathJax>
                    {" "}
                    {
                      "Observando el gráfico de \\( f(x) \\), identificamos un máximo en \\( x=0 \\) y un mínimo en \\( x=2 \\). Esto sugiere que \\( f'(x) = 0 \\) en esos puntos."
                    }{" "}
                  </MathJax>
                  <MathJax>
                    {" "}
                    {"\\[ f'(x) = a(x)(x - 2) = a(x^2 - 2x). \\]"}{" "}
                  </MathJax>
                  <MathJax>
                    {" "}
                    {
                      "Como la derivada debe ser \\( f'(x) = 3x^2 - 6x \\), entonces \\( a = 3 \\):"
                    }{" "}
                  </MathJax>
                  <MathJax> {"\\[ f'(x) = 3x^2 - 6x. \\]"} </MathJax>
                  <MathJax>
                    {" "}
                    {"Integrando para recuperar \\( f(x) \\):"}{" "}
                  </MathJax>
                  <MathJax>
                    {" "}
                    {
                      "\\[ f(x) = \\int (3x^2 - 6x)\\,dx = x^3 - 3x^2 + C. \\]"
                    }{" "}
                  </MathJax>
                  <MathJax>
                    {" "}
                    {
                      "Como en el gráfico se observa \\( f(0) = 2 \\), entonces \\( C = 2 \\):"
                    }{" "}
                  </MathJax>
                  <MathJax> {"\\[ f(x) = x^3 - 3x^2 + 2. \\]"} </MathJax>
                </div>
              </div>
            )}

            {answered && correct && (
              <div className="feedback-box correct">
                <i className="bi bi-check-circle"></i> ¡Correcto! Has deducido
                correctamente las dimensiones óptimas.
              </div>
            )}

            {answered && (
              <Button
                className="next-button"
                onClick={() => {
                  if (onNext) onNext(); // Llama a la función onNext si está definida
                  navigate("/analisis"); // Luego navega a /analisis
                }}
              >
                Feedback<i className="bi bi-arrow-right"></i>
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
              <strong>
                Tienen hasta <span style={{ color: "#007bff" }}>30 turnos</span>{" "}
                para llegar juntos a la respuesta final.
              </strong>
            </p>
            <p className="hint">
              <span style={{ color: "#dc3545", fontWeight: "bold" }}>
                Usa el botón <strong>“Llegamos a un acuerdo”</strong>{" "}
                <u>
                  solo si ambos ya están de acuerdo y quieren responder antes de
                  los 30 turnos
                </u>
                .
              </span>
            </p>
            <p className="hint">
              Apóyense usando el gráfico interactivo para visualizar ideas y
              validar sus propuestas.
            </p>
            <p className="hint" style={{ marginBottom: 0 }}>
              Consejos para colaborar bien:
            </p>
          </div>
          <ol className="collaboration-steps">
            <li>
              🤝 <strong>Escuchen y comprendan:</strong> Aclaren dudas y
              asegúrense de que ambos entienden el problema.
            </li>
            <li>
              🧠 <strong>Propongan y discutan ideas:</strong> Compartan puntos
              de vista, usen el gráfico y argumenten sus propuestas.
            </li>
            <li>
              ✅ <strong>Decidan juntos:</strong> Lleguen a una solución en
              conjunto y asegúrense de que ambos estén de acuerdo antes de
              responder.
            </li>
          </ol>
        </div>
      )}
    </div>
  );
};

export default Exercise2;
