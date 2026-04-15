import React, { useState } from "react";
import "./Header.css";
import logo from "../../assets/logo.png";
import "bootstrap/dist/css/bootstrap.min.css";

const Header = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="header fixed-top">
      <nav className="navbar navbar-expand-lg">
        <div className="container">
          <a
            className="navbar-brand fw-bold d-flex align-items-center"
            href="/"
          >
            <img src={logo} alt="Logo" className="brand-logo me-2" />
            <span className="brand-text">CollabMath</span>
          </a>

          <button
            className="navbar-toggler custom-toggler"
            type="button"
            aria-label="Toggle navigation"
            onClick={() => setIsOpen((prev) => !prev)}
          >
            <span></span>
            <span></span>
            <span></span>
          </button>

          <div
            className={`collapse navbar-collapse${isOpen ? " show" : ""}`}
            id="navbarNav"
          >
            <ul className="navbar-nav ms-auto">
              <li className="nav-item">
                <a className="nav-link nav-button" href="/register">
                  <i className="bi bi-person-plus me-1"></i>
                  Registrarse
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link nav-button" href="/login">
                  <i className="bi bi-box-arrow-in-right me-1"></i>
                  Iniciar Sesión
                </a>
              </li>
              <li className="nav-item">
                <a
                  className="nav-link nav-button admin-button"
                  href="/loginAdmin"
                >
                  <i className="bi bi-shield-lock me-1"></i>
                  Admin
                </a>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;
