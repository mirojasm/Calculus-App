import React, { useState, useEffect, useRef } from "react";
import { sendMessage } from "../../services/chatService";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Chat.css";
import FormulaEditor from "../FormulaEditor/FormulaEditor";
import { MathJaxContext, MathJax } from "better-react-mathjax";

const mathJaxConfig = {
  loader: { load: ["[tex]/ams"] },
  tex: {
    packages: { "[+]": ["ams"] },
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
  },
};

const Message = React.memo(({ msg }) => (
  <div
    className={`message ${
      msg.sender === "user" ? "user-message" : "bot-message"
    }`}
  >
    <div className="message-header">
      <i
        className={`bi ${
          msg.sender === "user" ? "bi-person-circle" : "bi-robot"
        }`}
      ></i>
      {msg.sender === "user" ? "Tú" : "Compañero"}
    </div>
    <div className="bubble">
      <MathJax>{msg.text}</MathJax>
    </div>
  </div>
));

const Chat = ({ currentExercise, onCollaborationPassed, PHASE_REACHED }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState(`conv_${Date.now()}`);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [collaborationCompleted, setCollaborationCompleted] = useState(false);
  const [showFormulaInput, setShowFormulaInput] = useState(false);
  const [formulaText, setFormulaText] = useState("");
  const [showAgreementButton, setShowAgreementButton] = useState(false);
  const [showAgreementModal, setShowAgreementModal] = useState(false);
  const chatEndRef = useRef(null);
  const mainInputRef = useRef(null);

  useEffect(() => {
    let initialText = "¡Hola! Soy tu compañero de cálculo.";
    if (currentExercise === 1) {
      initialText +=
        " Colaboremos juntos sobre este ejercicio de optimización multizona.";
    } else if (currentExercise === 2) {
      initialText +=
        " Colaboremos juntos sobre este ejercicio de función desconocida.";
    } else if (currentExercise === 3) {
      initialText = " Ejercicios completados.";
    }
    const initialMessages = [{ sender: "bot", text: initialText }];
    setMessages(initialMessages);
    setConversationId(`conv_${Date.now()}`);
    if (currentExercise === 3) {
      setCollaborationCompleted(true);
    } else {
      setCollaborationCompleted(false);
    }
    setInput("");
    setShowAgreementButton(false); // Oculta el botón al cambiar de ejercicio
  }, [currentExercise]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const timer = setTimeout(() => setShowAgreementButton(true), 10);
    return () => clearTimeout(timer);
  }, [currentExercise]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || collaborationCompleted) return;

    try {
      const userMessage = { sender: "user", text: input };
      setMessages((prev) => [...prev, userMessage]);
      setInput("");
      setLoading(true);
      setError("");

      const response = await sendMessage(
        input,
        conversationId,
        currentExercise
      );
      const botMessage = { sender: "bot", text: response.response };
      setMessages((prev) => [...prev, botMessage]);
      setConversationId(response.conversationId);
      if (response.collaborationPassed) {
        onCollaborationPassed();
        setCollaborationCompleted(true);
        setMessages((prev) => [
          ...prev,
          {
            sender: "bot",
            text: "✅ ¡Bien hecho! Has cumplido con la colaboración necesaria para pasar al siguiente nivel. Selecciona la alternativa correcta!!! ",
          },
        ]);
      }

      if (response.PHASE_REACHED) {
        PHASE_REACHED();
      }
    } catch (err) {
      setError("Error al enviar el mensaje. Intenta nuevamente.");
    } finally {
      setLoading(false);
    }
  };

  // Nueva función para manejar el botón de acuerdo
  const handleAgreement = () => {
    onCollaborationPassed();
    setCollaborationCompleted(true);
    setMessages((prev) => [
      ...prev,
      {
        sender: "bot",
        text: "✅ ¡Bien hecho! Has cumplido con la colaboración necesaria para pasar al siguiente nivel. Selecciona la alternativa correcta!!! ",
      },
    ]);
  };

  const handleAgreementClick = () => {
    setShowAgreementModal(true);
  };

  const handleAgreementConfirm = () => {
    setShowAgreementModal(false);
    handleAgreement();
  };

  const handleAgreementCancel = () => {
    setShowAgreementModal(false);
  };

  return (
    <MathJaxContext version={3} config={mathJaxConfig}>
      <div className="chat-wrapper">
        <div className="card-header">
          <i className="bi bi-robot me-2"></i> Compañero Virtual
        </div>

        <div className="chat-container">
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <Message key={index} msg={msg} />
            ))}

            {loading && (
              <div className="bot-message">
                <div className="message-header">
                  <i className="bi bi-robot"></i>Compañero
                </div>
                <div className="bubble d-flex align-items-center">
                  <div
                    className="spinner-border spinner-border-sm text-secondary me-2"
                    role="status"
                  ></div>
                  Escribiendo...
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        </div>

        <div className="card-footer">
          <form
            onSubmit={handleSendMessage}
            className="d-flex flex-column gap-2"
          >
            <div className="d-flex align-items-center gap-2">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setShowFormulaInput((prev) => !prev)}
                disabled={loading || collaborationCompleted}
              >
                ∑ Fórmula
              </button>

              <input
                type="text"
                ref={mainInputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  collaborationCompleted
                    ? "Has completado este ejercicio."
                    : "Escribe tu mensaje..."
                }
                className="form-control"
                disabled={loading || collaborationCompleted}
              />

              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || collaborationCompleted}
              >
                <i className="bi bi-send"></i>
              </button>
            </div>

            {showFormulaInput && (
              <FormulaEditor
                onInsert={(latex) => {
                  setInput((prev) => prev + ` $${latex}$ `);
                  setShowFormulaInput(false);
                  mainInputRef.current?.focus();
                }}
              />
            )}

            {/* Botón para indicar acuerdo */}
            {!collaborationCompleted && showAgreementButton && (
              <button
                type="button"
                className="btn btn-danger btn-sm mt-2"
                onClick={handleAgreementClick}
                disabled={loading}
              >
                <i className="bi bi-emoji-smile me-2"></i>
                Llegamos a un acuerdo
              </button>
            )}

            {/* Modal de advertencia */}
            {showAgreementModal && (
              <div
                className="modal show d-block"
                tabIndex="-1"
                style={{ background: "rgba(0,0,0,0.5)" }}
              >
                <div className="modal-dialog">
                  <div className="modal-content">
                    <div className="modal-header">
                      <h5 className="modal-title">¿Estás seguro?</h5>
                    </div>
                    <div className="modal-body">
                      <p>
                        Si confirmas que llegaron a un acuerdo, se desplegarán
                        las alternativas y <b>no podrás volver atrás</b>.
                      </p>
                    </div>
                    <div className="modal-footer">
                      <button
                        className="btn btn-secondary"
                        onClick={handleAgreementCancel}
                      >
                        Cancelar
                      </button>
                      <button
                        className="btn btn-danger"
                        onClick={handleAgreementConfirm}
                      >
                        Sí, confirmar
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {error && <p className="text-danger m-0">{error}</p>}
          </form>
        </div>
      </div>
    </MathJaxContext>
  );
};

export default Chat;
