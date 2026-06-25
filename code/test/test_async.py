import asyncio
import time

# Tạo một hàng đợi để chứa các đơn hàng
danh_sach_don_hang = asyncio.Queue()

# LUỒNG NGẦM (CONSUMER): Nhà bếp chuyên đợi và làm món
async def nha_bep_consumer():
    print("[Bếp] Đầu bếp đã sẵn sàng, ngồi đợi đơn hàng...")
    while True:
        # Đợi cho đến khi có đơn hàng trong hàng đợi
        mon_an = await danh_sach_don_hang.get()
        print(f"👉 [Bếp] Nhận đơn: Bắt đầu nấu món '{mon_an}'...")
        
        # Mô phỏng thời gian nấu món mất 2 giây
        # Trong 2 giây này, CPU sẽ rảnh và nhảy sang luồng nhận đơn!
        await asyncio.sleep(2) 
        
        print(f"✅ [Bếp] Hoàn thành: Món '{mon_an}' đã nấu xong!")
        # Báo cáo đã xử lý xong 1 đơn
        danh_sach_don_hang.task_done()

# LUỒNG CHÍNH (PRODUCER): Nơi nhận đơn từ khách hàng
async def main():
    # Kích hoạt luồng Nhà Bếp chạy ngầm
    nhiem_vu_bep = asyncio.create_task(nha_bep_consumer())
    
    # Giả sử có 3 khách hàng đặt món liên tiếp
    cac_mon_an = ["Phở Bò", "Bún Chả", "Cơm Tấm"]
    
    for mon in cac_mon_an:
        # breakpoint()  # <--- CHƯƠNG TRÌNH SẼ DỪNG LẠI TẠI ĐÂY
        print(f"🛒 [Khách] Đã đặt món: {mon}")
        # Bỏ món ăn vào hàng đợi để Bếp nhìn thấy
        await danh_sach_don_hang.put(mon)
        
        # Cứ 1 giây lại có 1 khách đặt món mới
        await asyncio.sleep(1)

    # Đợi cho đến khi hàng đợi trống (Bếp làm xong hết các món)
    await danh_sach_don_hang.join()
    
    # Hủy luồng nhà bếp vì không còn khách nào nữa
    nhiem_vu_bep.cancel()
    print("🏁 [Hệ thống] Đóng cửa hàng!")

# Chạy chương trình
asyncio.run(main())