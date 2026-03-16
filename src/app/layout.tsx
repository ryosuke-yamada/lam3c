import { Outlet } from "react-router";
import { Footer } from "@/components/footer";
import { Header } from "@/components/header";

export default function SharedUILayout() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background from-20% via-background-gradation-1 via-50% to-background-gradation-2 to-90%">
      <div className="bg-animated simple-grid flex min-h-screen w-full flex-col  items-center overflow-x-hidden">
        <Header />
        <div className="grow pt-16">
          <Outlet />
        </div>
        <Footer />
      </div>
    </div>
  );
}
