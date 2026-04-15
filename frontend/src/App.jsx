import { useState } from "react";
import { useNavigate } from "react-router-dom"; // Import useNavigate
import logo from "./assets/logo.png"; // Adjust the path as necessary
import "bootstrap/dist/css/bootstrap.min.css";
import Header from "./components/Header/Header"; // Assuming the header is in this path

function App() {
  const navigate = useNavigate(); // Initialize useNavigate

  const handleRegister = () => {
    navigate("/register"); // Navigate to the register page
  };

  const handleLogin = () => {
    navigate("/login"); // Navigate to the login page
  };

  const consentSections = [
    {
      title: "Puntos Importantes",
      icon: "⚡",
      content:
        "•Es recomendable usar la aplicación en un <b>computador</b>, ya que está mejor optimizada para ese formato.<br /> •Es necesario contar con <b>conexión a internet</b> para su correcto funcionamiento.<br />• Es clave que completes el formulario al final de la sesión, ya que es parte fundamental del estudio.",
    },
    {
      title: "Objetivo del Estudio",
      icon: "🎯",
      content:
        "Diseñar, implementar y estudiar empíricamente la integración de inteligencia artificial generativa en entornos educativos para mejorar la autenticidad, validez y escalabilidad en la medición y desarrollo de habilidades de colaboración, comunicación, creatividad y pensamiento crítico.",
    },
    {
      title: "¿Por qué fue seleccionado?",
      icon: "👥",
      content:
        "Usted ha sido seleccionado para participar debido a su pertenencia a la población de estudio, que incluye estudiantes de Ingeniería Civil.",
    },
    {
      title: "Procedimientos",
      icon: "📋",
      content:
        "Su participación consiste en realizar de manera individual una serie de actividades digitales de evaluación y desarrollo de habilidades del siglo XXI, en el contexto del contenido visto en su carrera sobre cálculo diferencial. La sesión tendrá una duración aproximada de 45 minutos.",
    },
    {
      title: "Beneficios",
      icon: "✨",
      content:
        "Aunque no obtendrá beneficios personales directos, la información que proporcione será de gran valor para mejorar las prácticas educativas y beneficiar a otros estudiantes en el futuro.",
    },
    {
      title: "Confidencialidad",
      icon: "🔒",
      content:
        "Su participación será completamente anónima. Sus datos personales y respuestas no serán asociados a su nombre. La información será usada exclusivamente para propósitos académicos y almacenada de forma segura por 5 años.",
    },
    {
      title: "Voluntariedad",
      icon: "🤝",
      content:
        "Su participación es completamente voluntaria. Tiene derecho a no participar o retirar su consentimiento en cualquier momento, sin explicaciones ni consecuencias.",
    },
  ];

  return (
    <div
      className="min-vh-100 d-flex flex-column"
      style={{ backgroundColor: "#f8f9fa" }}
    >
      <Header />

      <main className="container py-5">
        <div className="row justify-content-center">
          <div className="col-lg-10 col-xl-8">
            <div className="text-center mb-5">
              <div className="logo-container mb-4">
                <div
                  className="logo-placeholder bg-primary rounded-circle mx-auto d-flex align-items-center justify-content-center text-white"
                  style={{ width: "120px", height: "120px", fontSize: "3rem" }}
                >
                  <img
                    src={logo}
                    alt="Logo"
                    style={{
                      width: "80px",
                      height: "80px",
                      objectFit: "contain",
                    }}
                  />
                </div>
              </div>
              <h1 className="display-5 fw-bold text-primary mb-3">
                Aplicación de Cálculo Diferencial
              </h1>
              <p className="lead text-muted">
                Investigación en Educación con Inteligencia Artificial
              </p>
            </div>

            <div className="card shadow-lg border-0 mb-5">
              <div
                className="card-header bg-gradient text-white text-center py-4"
                style={{
                  background: "linear-gradient(135deg, #007bff, #0056b3)",
                }}
              >
                <h2 className="h3 mb-0">📋 Consentimiento Informado</h2>
                <p className="mb-0 opacity-75">
                  Por favor, lea cuidadosamente la siguiente información
                </p>
              </div>

              <div className="card-body p-0">
                {consentSections.map((section, index) => (
                  <div
                    key={index}
                    className={`p-4 ${
                      index < consentSections.length - 1 ? "border-bottom" : ""
                    }`}
                  >
                    <div className="d-flex align-items-start">
                      <div className="flex-shrink-0 me-3">
                        <div
                          className="bg-light rounded-circle p-2 d-flex align-items-center justify-content-center"
                          style={{
                            width: "40px",
                            height: "40px",
                            fontSize: "1.2rem",
                          }}
                        >
                          {section.icon}
                        </div>
                      </div>
                      <div className="flex-grow-1">
                        <h3 className="h5 fw-bold text-dark mb-2">
                          {section.title}
                        </h3>
                        <p
                          className="text-muted mb-0 lh-base"
                          dangerouslySetInnerHTML={{ __html: section.content }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="text-center">
              <div className="d-flex flex-column flex-sm-row justify-content-center gap-3">
                <button
                  className="btn btn-primary btn-lg px-5 py-3 shadow-sm fw-semibold"
                  onClick={handleRegister}
                  style={{
                    borderRadius: "50px",
                    transition: "all 0.3s ease",
                    border: "none",
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow = "0 8px 25px rgba(0,123,255,0.3)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow = "0 2px 10px rgba(0,0,0,0.1)";
                  }}
                >
                  ✅ Acepto y me Registro
                </button>
                <button
                  className="btn btn-outline-secondary btn-lg px-5 py-3 shadow-sm fw-semibold"
                  onClick={handleLogin}
                  style={{
                    borderRadius: "50px",
                    transition: "all 0.3s ease",
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow =
                      "0 8px 25px rgba(108,117,125,0.3)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow = "0 2px 10px rgba(0,0,0,0.1)";
                  }}
                >
                  🔑 Iniciar Sesión
                </button>
              </div>
              <p className="text-muted mt-3 small">
                Al hacer clic en "Acepto y me Registro", confirma que ha leído y
                acepta participar en este estudio
              </p>
            </div>
          </div>
        </div>
      </main>

      <style jsx>{`
        .bg-gradient {
          background: linear-gradient(135deg, #007bff, #0056b3) !important;
        }

        .logo-placeholder {
          transition: transform 0.3s ease;
        }

        .logo-placeholder:hover {
          transform: scale(1.05);
        }

        .card {
          transition: transform 0.2s ease;
        }

        .card:hover {
          transform: translateY(-2px);
        }

        .btn {
          font-weight: 600;
          letter-spacing: 0.5px;
        }

        .text-primary {
          color: #007bff !important;
        }

        .lh-base {
          line-height: 1.6;
        }
      `}</style>
    </div>
  );
}

export default App;
