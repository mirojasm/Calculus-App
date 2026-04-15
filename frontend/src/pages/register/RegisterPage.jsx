import React, { useState } from "react";
import { register } from "../../services/registerService";
import { useNavigate } from "react-router-dom";
import "./RegisterPage.css";
import logo2 from "../../assets/logo2.png";
import Header from "../../components/Header/Header";

const RegisterPage = () => {
  const [nombre, setNombre] = useState("");
  const [correo, setCorreo] = useState("");
  const [nickname, setNickname] = useState("");
  const [contrasena, setContrasena] = useState("");
  const [repetirContrasena, setRepetirContrasena] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showRepeatPassword, setShowRepeatPassword] = useState(false);
  const [success, setSuccess] = useState("");
  const [error, setError] = useState("");

  const navigate = useNavigate();
  const passwordsMatch = contrasena === repetirContrasena;

  const handleRegister = async (e) => {
    e.preventDefault();
    if (!passwordsMatch) {
      setError("Las contraseñas no coinciden.");
      setSuccess("");
      return;
    }
    try {
      const userData = await register(nombre, correo, nickname, contrasena);
      //console.log("Usuario registrado:", userData);
      //setSuccess("Registro exitoso. Ahora puedes iniciar sesión.");
      setError("");
      navigate("/login");
    } catch (err) {
      console.error("Error en el registro:", err);
      let errorMsg = "Error en el registro. Intenta nuevamente.";
      const backendError = err?.response?.data?.error;
      if (backendError) {
        if (backendError.includes('usuarios_nickname_key')) {
          errorMsg = "El nombre de usuario ya está registrado.";
        } else if (backendError.includes('usuarios_correo_key')) {
          errorMsg = "El correo ya está registrado.";
        }
      }
      setError(errorMsg);
      setSuccess("");
    }
  };

  return (
    <>
      <Header />
      <div className="register-wrapper">
        <div className="register-form fade-in">
          <div className="logo-container">
            <img src={logo2} alt="Logo" className="register-logo" />
          </div>
          <h2 className="register-title">Registro</h2>

          <form onSubmit={handleRegister}>
            <div className="form-row">
              <div className="form-group half-width">
                <label htmlFor="nombre">Nombre y Apellido</label>
                <input
                  id="nombre"
                  type="text"
                  value={nombre}
                  onChange={(e) => setNombre(e.target.value)}
                  required
                  placeholder="Ingresa tu nombre"
                />
              </div>

              <div className="form-group half-width">
                <label htmlFor="correo">Correo electrónico</label>
                <input
                  id="correo"
                  type="email"
                  value={correo}
                  onChange={(e) => setCorreo(e.target.value)}
                  required
                  placeholder="Ingresa tu correo"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group half-width">
                <label htmlFor="nickname">Username</label>
                <input
                  id="nickname"
                  type="text"
                  value={nickname}
                  onChange={(e) => setNickname(e.target.value)}
                  required
                  placeholder="Ingresa tu username"
                />
              </div>

              <div className="form-group half-width">
                <label htmlFor="contrasena">Contraseña</label>
                <div className="password-input-container">
                  <input
                    id="contrasena"
                    type={showPassword ? "text" : "password"}
                    value={contrasena}
                    onChange={(e) => setContrasena(e.target.value)}
                    required
                    placeholder="Ingresa tu contraseña"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="password-toggle"
                    tabIndex={-1}
                  >
                    <i className={`bi ${showPassword ? "bi-eye-slash" : "bi-eye"}`}></i>
                  </button>
                </div>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="repetirContrasena">Repetir Contraseña</label>
              <div className="password-input-container">
                <input
                  id="repetirContrasena"
                  type={showRepeatPassword ? "text" : "password"}
                  value={repetirContrasena}
                  onChange={(e) => setRepetirContrasena(e.target.value)}
                  required
                  placeholder="Repite tu contraseña"
                  className={!passwordsMatch && repetirContrasena ? "invalid" : ""}
                />
                <button
                  type="button"
                  onClick={() => setShowRepeatPassword(!showRepeatPassword)}
                  className="password-toggle"
                  tabIndex={-1}
                >
                  <i className={`bi ${showRepeatPassword ? "bi-eye-slash" : "bi-eye"}`}></i>
                </button>
              </div>
              {!passwordsMatch && repetirContrasena && (
                <div className="field-error">Las contraseñas no coinciden.</div>
              )}
            </div>

            {success && <div className="error-message" style={{background: '#f0fff4', color: '#38a169', borderColor: '#9ae6b4'}}>{success}</div>}
            {error && <div className="error-message">{error}</div>}

            <button type="submit" className="register-button">
              Registrarse
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

export default RegisterPage;