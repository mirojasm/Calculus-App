import "./index.css";
import App from "./App.jsx";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import RegisterPage from "./pages/register/RegisterPage.jsx";
import LoginPage from "./pages/Login/LoginPage.jsx";
import Chat from "./components/Chat/Chat.jsx";
import LoginAdminPage from "./pages/loginAdmin/LoginAdminPage.jsx";
import ConversationsPage from "./pages/Conversation/ConversationsPage.jsx";
import Principal from "./pages/Principal/Principal.jsx";
import Analisis from "./pages/Analisis/Analisis.jsx";
import UsuariosPage from "./pages/Usuarios/UsuariosPage.jsx";
import AnalisisPage from "./pages/AnalisisAdmin/AnalisisPage.jsx";
import ResultadosPage from "./pages/ResultadosPage/ResultadosPage.jsx";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <Router>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/principal" element={<Principal />} />
        <Route path="/loginAdmin" element={<LoginAdminPage />} />
        <Route path="/conversaciones" element={<ConversationsPage />} />
        <Route path="/usuarios" element={<UsuariosPage/>}/>
        <Route path="/analisis" element={<Analisis/>} />
        <Route path="/analisisAdmin" element={<AnalisisPage/>} />
        <Route path="/resultados" element={<ResultadosPage/>} />
      </Routes>
    </Router>
  </React.StrictMode>
);
