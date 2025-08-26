// frontend/react_app/src/components/common/Layout.jsx
import React from "react";
import { useLocation } from "react-router-dom";
import Header from "./Header";
import Sidebar from "./Sidebar";
import "../../styles/components/Layout.css";

const Layout = ({ children }) => {
  const location = useLocation();

  return (
    <div className="layout">
      <Header />
      <div className="layout-content">
        <Sidebar currentPath={location.pathname} />
        <main className="main-content">{children}</main>
      </div>
    </div>
  );
};

export default Layout;
