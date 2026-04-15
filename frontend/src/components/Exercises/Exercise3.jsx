import React, { useState, useEffect } from "react";
import { Button } from "react-bootstrap";
import "./Exercise.css";

const Exercise3 = ({ onNext }) => {
  const [collabUnlocked, setCollabUnlocked] = useState(false);
  const [answered, setAnswered] = useState(false);
  const [correct, setCorrect] = useState(null);
  const [showNext, setShowNext] = useState(false);

  const handleAnswer = (choice) => {
    const isCorrect = choice === "C";
    setAnswered(true);
    setCorrect(isCorrect);
    setShowNext(true);
  };

  useEffect(() => {
    const handleEvent = () => setCollabUnlocked(true);
    window.addEventListener("collab-unlocked-1", handleEvent);
    
    return () => {
      window.removeEventListener("collab-unlocked-1", handleEvent);
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
        <h3>Derivada de una función trigonométrica</h3>
        <p className="mb-4">
          ¿Cuál es la derivada de la función \( f(x) = \sin(x) + x^2 \)?
        </p>
    
        <div className="geogebra-container mb-4">
      <iframe
  title="GeoGebra Interactivo"
  src="https://www.geogebra.org/material/iframe/id/jw7dua33/width/800/height/450/rc/true/ai/true/sdz/true"
  loading="lazy"
  width="100%"
  height="500px"
  style={{ border: "1px solid #ccc", borderRadius: "10px" }}
  allowFullScreen
></iframe>
</div>
      </div>

      {collabUnlocked ? (
        <div className="exercise-options">
          <Button
            variant={getVariant("A")}
            className="option-button"
            onClick={() => handleAnswer("A")}
            disabled={answered}
          >
            A) \( \cos(x) \)
          </Button>
          
          <Button
            variant={getVariant("B")}
            className="option-button"
            onClick={() => handleAnswer("B")}
            disabled={answered}
          >
            B) \( \cos(x) + 2 \)
          </Button>
          
          <Button
            variant={getVariant("C")}
            className="option-button"
            onClick={() => handleAnswer("C")}
            disabled={answered}
          >
            C) \( \cos(x) + 2x \)
          </Button>

          {answered && !correct && (
            <div className="feedback-box incorrect">
              <i className="bi bi-x-circle"></i> Incorrecto. La derivada de \(\sin(x)\) es \(\cos(x)\) y la derivada de \(x^2\) es \(2x\).
            </div>
          )}

          {answered && correct && (
            <div className="feedback-box correct">
              <i className="bi bi-check-circle"></i> ¡Correcto! Has derivado correctamente cada término.
            </div>
          )}

          {answered && (
            <Button className="next-button" onClick={onNext}>
              Siguiente ejercicio <i className="bi bi-arrow-right"></i>
            </Button>
          )}
        </div>
      ) : (
        <div className="collaboration-message">
          <i className="bi bi-chat-dots"></i>
          <p>Colabora con el asistente virtual antes de continuar...</p>
          <p className="hint">Pregunta sobre cómo derivar funciones como \( \sin(x) \) y \( x^2 \)</p>
        </div>
      )}
    </div>
  );
};

export default Exercise3;

