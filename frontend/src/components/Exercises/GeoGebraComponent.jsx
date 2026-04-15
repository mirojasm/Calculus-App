import React, { useEffect } from "react";

const GeoGebraComponent = () => {
  useEffect(() => {
    const params = {
      id: "ggbApplet",
      appName: "graphing",
      material_id: "gm7v6ctw",
      width: 700,
      height: 400,
      showToolBar: false,
      showMenuBar: false,
      showAlgebraInput: false,
      enableShiftDragZoom: true,
      enableRightClick: false,
      useBrowserForJS: true,
    };
    const applet = new GGBApplet(params, true);
    applet.inject("ggb-element"); 

    
  }, []);

  return <div id="ggb-element" />;
};

export default GeoGebraComponent;
