import React, { useState, useEffect, useRef } from "react";
import { analisisService } from "../../services/analisisService";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Analisis.css";
import logo from "../../assets/logo.png";

const parseAnalisisToHtml = (text) => {
  if (!text) return "";
  // Títulos tipo ### Título
  let html = text.replace(
    /^### (.*)$/gm,
    '<h4 class="mt-4 mb-2 fw-bold">$1</h4>'
  );
  // Negritas tipo **texto**
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // Saltos de línea simples a <br>
  html = html.replace(/\n/g, "<br />");

  return html;
};

const Analisis = () => {
  const [nickname, setNickname] = useState("");
  const [analisis, setAnalisis] = useState("");
  const [loading, setLoading] = useState(true);

  // Ref para evitar múltiples llamadas
  const hasFetched = useRef(false);

  useEffect(() => {
    const userData = JSON.parse(localStorage.getItem("user"));
    if (userData && userData.nickname) {
      setNickname(userData.nickname);
    }
  }, []);

  useEffect(() => {
    // Evitar múltiples llamadas usando useRef
    if (hasFetched.current) return;

    const fetchAnalisis = async () => {
      hasFetched.current = true; // Marcar como ejecutado
      try {
        console.log("Ejecutando fetchAnalisis - una sola vez");
        const data = await analisisService();
        if (data && data.analisis) {
          setAnalisis(data.analisis);
        }
      } catch (error) {
        console.error("Error al obtener el análisis:", error);
        hasFetched.current = false; // Permitir reintentos en caso de error
      } finally {
        setLoading(false);
      }
    };

    fetchAnalisis();
  }, []);

  return (
    <>
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

      <main className="container py-5">
        <div className="row justify-content-center">
          <div className="col-lg-10 col-xl-8">
            {/* Mensaje importante para la tesis */}
            <div
              className="alert alert-primary text-center fw-bold mb-3"
              role="alert"
              style={{ fontSize: "1.15rem" }}
            >
              Por favor responde el formulario del final cuando el feedback este
              cargado, es muy importante para mi tesis.
            </div>

            <div className="card shadow-lg border-0 mb-5">
              <div
                className="card-header bg-gradient text-white text-center py-4"
                style={{
                  background: "linear-gradient(135deg, #007bff, #0056b3)",
                }}
              >
                <h2 className="h3 mb-0">📊 Feedback</h2>
                <p className="mb-0 opacity-75">
                  Resultado de tu feedback personalizado
                </p>
              </div>
              <div className="card-body p-4">
                {loading ? (
                  <div className="text-center py-5">
                    <div
                      className="mb-3 fw-semibold text-primary"
                      style={{ fontSize: "1.2rem" }}
                    >
                      Cargando Feedback...
                    </div>
                    <div className="spinner-border text-primary" role="status">
                      <span className="visually-hidden">
                        Cargando Feedback...
                      </span>
                    </div>
                  </div>
                ) : analisis ? (
                  <div
                    className="text-dark lh-base"
                    style={{ whiteSpace: "normal" }}
                    dangerouslySetInnerHTML={{
                      __html: parseAnalisisToHtml(analisis),
                    }}
                  />
                ) : (
                  <p className="text-muted text-center mb-0">
                    No se encontró análisis para mostrar.
                  </p>
                )}
              </div>
            </div>

            {/* Nueva tarjeta para el formulario */}
            <div className="card shadow border-0 ">
              <div
                className="card-header bg-gradient text-white text-center py-3"
                style={{
                  background: "linear-gradient(135deg, #007bff, #0056b3)",
                }}
              >
                <h3 className="h5 mb-0">📑 Por favor responde el formulario</h3>
              </div>
              <div className="card-body text-center">
                <p className="mb-4 text-dark">
                  Tu respuesta es muy importante para mi tesis. ¡Gracias por tu
                  colaboración!
                </p>
                <a
                  href="https://docs.google.com/forms/d/e/1FAIpQLScFy5xnz6CHBWgKU8wwdfwVJYdWYdW7mFxlzPRshNDcicysRQ/viewform?usp=dialog"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-primary btn-lg px-5 py-3 fw-semibold"
                  style={{ borderRadius: "50px" }}
                >
                  Ir al Formulario de Google
                </a>
              </div>
            </div>
          </div>
        </div>
      </main>

      <div className="text-center my-5">
        <h2
          style={{
            fontWeight: 700,
            fontSize: "2.1rem",
            color: "#007bff",
            letterSpacing: "0.5px",
          }}
        >
          🎉 ¡Gracias por darte el tiempo de probar el software!
          <br />
        </h2>
      </div>

      <style jsx>{`
        .bg-gradient {
          background: linear-gradient(135deg, #007bff, #0056b3) !important;
        }
        .lh-base {
          line-height: 1.6;
        }
        .card-body h4 {
          color: #0056b3;
        }
      `}</style>
    </>
  );
};

export default Analisis;
