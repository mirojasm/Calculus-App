import React, { useState, useEffect } from "react";
import Chat from "../../components/Chat/Chat";
import { useNavigate } from "react-router-dom";
import { Button } from "react-bootstrap";
import logo from "../../assets/logo.png";
import { accionService } from "../../services/accionService";
import Exercise1 from "../../components/Exercises/Exercise1";
import Exercise2 from "../../components/Exercises/Exercise2";
import Exercise3 from "../../components/Exercises/Exercise3";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Principal.css";

const Principal = () => {
  const navigate = useNavigate();
  const [nickname, setNickname] = useState("");

  // ✅ Inicializar desde localStorage antes del primer render
  const [currentExercise, setCurrentExercise] = useState(() => {
    //localStorage.removeItem("currentExercise");
    const saved = localStorage.getItem("currentExercise");
    return saved ? parseInt(saved, 10) : 1;
  });

  const currentExerciseVista =
    currentExercise === 3
      ? "Ejercicios Completados"
      : `Ejercicio ${currentExercise}`;

  const [progressStage, setProgressStage] = useState(0);
  const collabScore = progressStage * 25;

  // ✅ Guardar currentExercise en localStorage cuando cambia
  useEffect(() => {
    localStorage.setItem("currentExercise", currentExercise);
  }, [currentExercise]);

  useEffect(() => {
    const userData = JSON.parse(localStorage.getItem("user"));
    if (userData && userData.nickname) {
      setNickname(userData.nickname);
    }
  }, []);
  // ✅ Registrar clics y zona (chatbot, gráfico, otro)
  useEffect(() => {
    const handleClick = async (event) => {
      if (event.target.closest(".chat-panel")) {
        const accion = "Click detectado en compañero virtual";
        const current_exercise = currentExercise;
        const accionData = await accionService(accion, current_exercise);
        console.log("accion registrada:", accionData);
      }
    };

    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

  const advanceStage = () => {
    console.log("✅ PHASE_REACHED recibido");
    setProgressStage((prev) => Math.min(prev + 1, 3));
  };

  const completeProgress = () => {
    console.log("🎉 onCollaborationPassed recibido - Completando barra");
    setTimeout(() => {
      setProgressStage(4);
      window.dispatchEvent(new CustomEvent(`collab-unlocked-1`));
    }, 100);
  };

  const goToNextExercise = () => {
    setProgressStage(0);

    setCurrentExercise((prev) => {
      const nextExercise = Math.min(prev + 1, 4);

      // ✅ Si se completó el último ejercicio, borrar localStorage
      if (prev === 1) {
        localStorage.removeItem("currentExercise");
      }

      return nextExercise;
    });
  };

  const renderExercise = () => {
    const props = { onNext: goToNextExercise };

    switch (currentExercise) {
      case 1:
        return <Exercise1 {...props} />;
      case 2:
        return <Exercise2 {...props} />;
      default:
        return (
          <>
            <h3 className="text-center mt-5">¡Ejercicios completados!</h3>
            <div className="d-flex justify-content-center">
              <Button
                className="next-button"
                onClick={() => {
                  navigate("/analisis"); // Luego navega a /analisis
                }}
              >
                Feedback <i className="bi bi-arrow-right"></i>
              </Button>
            </div>
          </>
        );
    }
  };

  return (
    <div className="app-container">
      <header className="header d-flex justify-content-between align-items-center px-3">
        <a className="navbar-brand fw-bold d-flex align-items-center">
          <img src={logo} alt="Logo" className="brand-logo me-2" />
          <span className="brand-text">CollabMath</span>
        </a>

        {nickname && (
          <div className="user-chip d-flex align-items-center ms-4">
            <i className="bi bi-person-circle me-2 fs-3"></i>
            <span className="nickname-badge px-4 py-2 rounded-pill">
              {nickname}
            </span>
          </div>
        )}
      </header>

      <main className="content-container">
        <section className="exercise-panel">
          <div className="exercise-header">
            <h2>{currentExerciseVista}</h2>
            <div className="collab-bar-container">
              <h5>Progreso de Colaboración</h5>
              <div className="progress">
                <div
                  className="progress-bar progress-bar-striped progress-bar-animated bg-success"
                  role="progressbar"
                  style={{
                    width: `${collabScore}%`,
                    minWidth: "3ch", // suficiente para mostrar "0%"
                  }}
                  aria-valuenow={collabScore}
                  aria-valuemin="0"
                  aria-valuemax="100"
                >
                  {collabScore}%
                </div>
              </div>
            </div>
          </div>

          <div className="exercise-content">{renderExercise()}</div>
        </section>

        <aside className="chat-panel">
          <Chat
            currentExercise={currentExercise}
            onCollaborationPassed={completeProgress}
            PHASE_REACHED={advanceStage}
          />
        </aside>
      </main>

      
    </div>
  );
};

export default Principal;
