import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../../services/loginService";
import "bootstrap/dist/css/bootstrap.min.css";
import "./LoginPage.css";
import logo2 from "../../assets/logo2.png";
import Header from "../../components/Header/Header";

const LoginPage = () => {
  const [nickname, setNickname] = useState("");
  const [contrasena, setContrasena] = useState("");
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const userData = await login(nickname, contrasena);
      //console.log("Usuario logueado:", userData);
      setError("");
      navigate("/principal");
    } catch (err) {
      console.error("Error de inicio de sesión:", err);
      setError("Credenciales incorrectas. Intenta nuevamente.");
    }
  };

  return (
    <>
      <Header />
      <div className="login-wrapper">
        <div className="login-form fade-in">
          <div className="logo-container">
            <img src={logo2} alt="Logo" className="login-logo" />
          </div>
          <h2 className="login-title">Iniciar Sesión</h2>
          
          <form onSubmit={handleLogin}>
            <div className="form-group">
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
            
            <div className="form-group">
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
                  className="password-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                  tabIndex={-1}
                >
                  <i className={`bi ${showPassword ? "bi-eye-slash" : "bi-eye"}`}></i>
                </button>
              </div>
            </div>
            
            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
            
            <button type="submit" className="login-button">
              Entrar
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

export default LoginPage;
