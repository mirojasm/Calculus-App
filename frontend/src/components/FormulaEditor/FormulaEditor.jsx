import React, { useEffect, useRef, useState } from "react";
import { MathfieldElement } from "mathlive";

const FormulaEditor = ({ onInsert }) => {
  const containerRef = useRef(null);
  const mathFieldRef = useRef(null);
  const [isTouchDevice, setIsTouchDevice] = useState(false);

  useEffect(() => {
    const mf = new MathfieldElement({
      smartMode: true,
      virtualKeyboardMode: "off", // Permite teclado nativo en móvil
      smartFence: true,
      // Configuraciones adicionales para eliminar el menú hamburguesa
      menuMode: "off", // Elimina el menú hamburguesa
      mathModeSpace: "\\:", // Evita espacios automáticos
      readOnly: false,
      // Elimina completamente la UI adicional
      virtualKeyboard: null,
      menu: null,
      // Configuraciones específicas para el teclado
      virtualKeyboards: "none",
      keypressVibration: false,
      keypressSound: null,
      plonkSound: null,
      // Desactiva completamente la funcionalidad del teclado virtual
      useSharedVirtualKeyboard: false,
      createHTML: (text) => text,
    });

    mf.style.minHeight = "60px";
    mf.style.maxHeight = "60px";
    mf.style.overflowY = "auto";
    mf.style.width = "100%";
    mf.classList.add("custom-mathfield");

    // Eliminar completamente cualquier elemento del menú o teclado
    mf.addEventListener("mount", () => {
      // Eliminar el listener que ocultaba el teclado en focus
      // mf.addEventListener("focus", (e) => {
      //   if (mf.virtualKeyboard) {
      //     mf.virtualKeyboard.hide();
      //   }
      // });

      mf.addEventListener("virtual-keyboard-toggle", (e) => {
        e.preventDefault();
        e.stopPropagation();
      });

      // Ocultar cualquier elemento de UI que pueda aparecer
      const shadowRoot = mf.shadowRoot;
      if (shadowRoot) {
        const style = document.createElement("style");
        style.textContent = `
          .ML__keyboard,
          .ML__virtual-keyboard,
          .ML__virtual-keyboard-toggle,
          .ML__menu,
          .ML__menu-button,
          .ML__menu-toggle,
          .ML__popover,
          .ML__tooltip,
          .ML__sr-only,
          math-field::part(virtual-keyboard),
          math-field::part(menu) {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
            position: absolute !important;
            left: -9999px !important;
            width: 0 !important;
            height: 0 !important;
          }
        `;
        shadowRoot.appendChild(style);
      }
    });

    mathFieldRef.current = mf;
    containerRef.current.innerHTML = "";
    containerRef.current.appendChild(mf);

    setIsTouchDevice(
      "ontouchstart" in window ||
        (window.matchMedia && window.matchMedia("(pointer: coarse)").matches)
    );
  }, []);

  const insertCommand = (command) => {
    mathFieldRef.current.focus();
    mathFieldRef.current.insert(command);
  };

  const handleInsertClick = () => {
    const latex = mathFieldRef.current.getValue();
    if (latex) {
      onInsert(latex);
      mathFieldRef.current.setValue("");
    }
  };

  return (
    <div className="formula-editor">
      {isTouchDevice && (
        <div style={{ color: '#007bff', fontSize: '0.98rem', marginBottom: 6, textAlign: 'center' }}>
          <b>mantén pulsado el campo</b> de abajo para desplegar el teclado de formulas.
        </div>
      )}
      {!isTouchDevice && (
        <div className="btn-toolbar mb-2" role="toolbar">
          <div className="btn-group me-2 mb-1" style={{ flexWrap: "wrap" }}>
            <button
              className="btn btn-primary"
              onClick={(e) => {
                e.preventDefault();
                insertCommand("f(x)");
              }}
            >
              f(x)
            </button>
            <button
              className="btn btn-primary"
              onClick={(e) => {
                e.preventDefault();
                insertCommand("\\cdot");
              }}
            >
              ·
            </button>
            <button
              className="btn btn-primary"
              onClick={(e) => {
                e.preventDefault();
                insertCommand("\\frac{#@}{□}");
              }}
            >
              /
            </button>
            <button
              className="btn btn-primary"
              onClick={(e) => {
                e.preventDefault();
                insertCommand("^2");
              }}
            >
              □²
            </button>
            <button
              className="btn btn-primary"
              onClick={(e) => {
                e.preventDefault();
                insertCommand("x^2");
              }}
            >
              x²
            </button>
            <button
              className="btn btn-primary"
              onClick={(e) => {
                e.preventDefault();
                insertCommand("\\pi");
              }}
            >
              π
            </button>
            <button
              className="btn btn-primary"
              onClick={(e) => {
                e.preventDefault();
                insertCommand("\\frac{d}{dx}");
              }}
            >
              d/dx
            </button>
            <button
              className="btn btn-primary"
              onClick={(e) => {
                e.preventDefault();
                insertCommand("\\int \\placeholder{} \\,dx");
              }}
            >
              ∫
            </button>
          </div>
        </div>
      )}
      <div
        style={{ display: "flex", alignItems: "center", gap: "10px" }}
        className="mb-2"
      >
        <div
          ref={containerRef}
          className="mathlive-input"
          style={{ flexGrow: 1, maxWidth: "600px", minWidth: "200px" }}
        ></div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={(e) => {
            e.preventDefault(); // ← evita el submit del formulario
            handleInsertClick();
          }}
        >
          Insertar
        </button>
      </div>
    </div>
  );
};

export default FormulaEditor;
