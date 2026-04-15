import React from "react";
import "./Header.css";
import logo from "../../assets/logo.png";
import "bootstrap/dist/css/bootstrap.min.css";

const HeaderAdmin = () => {
  return (
    <header className="header fixed-top">
      <nav className="navbar navbar-expand-lg">
        <div className="container">
          <a
            className="navbar-brand fw-bold d-flex align-items-center"
          >
            <img src={logo} alt="Logo" className="brand-logo me-2" />
            <span className="brand-text">CollabMath</span>
          </a>

          <button
            className="navbar-toggler custom-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarNav"
          >
            <span></span>
            <span></span>
            <span></span>
          </button>

          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav ms-auto">
              <li className="nav-item">
                <a className="nav-link nav-button" href="/usuarios">
                  <i className="bi bi-person-circle me-1"></i>
                  Usuarios
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link nav-button" href="/conversaciones">
                  <i className="bi bi-chat-dots me-1"></i>
                  Convesaciones
                </a>
              </li>
               <li className="nav-item">
                <a className="nav-link nav-button" href="/resultados">
                  <i className="bi bi-award me-1"></i>
                  Resultados
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link nav-button" href="/analisisAdmin">
                  <i className="bi bi-clipboard-data me-1"></i>
                  Analisis
                </a>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </header>
  );
};

export default HeaderAdmin;
