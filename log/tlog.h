#ifndef __LOG_H__
#define __LOG_H__

#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef WIN32
#ifdef __WINSOCK2__
#include <winsock2.h>
#else
#include <winsock.h>
#endif
#include <cassert>
#include <io.h>
#else
#include <unistd.h>
#include <assert.h>
#endif


#include "KernelObject.h"
#include "cttypes.h"

#define MAX_FILE_LINES	(200000)
#define MAX_FILE_SIZE	(20*1024*1024) //20M

#define LOG_TO_NULL		0
#define LOG_TO_FILE		1
#define LOG_TO_SCREEN	2
#define LOG_TO_SCNFILE	3
#define	LOG_TO_UNICODEFILE	4

#ifndef WIN32
#define MAX_PATH			256
#endif
#define MAX_VERSION_LEN		256


//////////////////////////////////////////////////////////////////////////
//日志事件处理接口
class ILogEventHandler
{
public:
	//////////////////////////////////////////////////////////////////////////
	//日志备份事件
	//@filename:文件名
	//@appName:应用名
	//@moduleName:模块名	
	//@beginTime:日志开始时间
	//@endTime:日志结束时间
	//@lines:日志行数
	virtual	void OnLogBackup(const CTstring& filename, const CTstring& appName, const CTstring& moduleName,
		const CTtime& beginTime, const CTtime& endTime, const CTuint lines)
	{
	}
};

typedef enum 
{
	LogLevel_Closed = 0,
	LogLevel_Fatal = 1,
	LogLevel_Error = 2,
	LogLevel_Warn = 3,
	LogLevel_Info = 4,
	LogLevel_Debug = 5,
}CTLogLevel;

typedef enum
{
	LogBackupT_None = 0, 
	LogBackupT_Quarter = 1,
	LogBackupT_Half = 2,
	LogBackupT_Hour = 3,	
	LogBackupT_Day = 4,
}CTLogBackupType;

class TLog
{
public:
	TLog(const char *szVersion);
	TLog(const char *module, const char *version, int multiuser = 1, const char *logPath = NULL);
	TLog(const char *appname, const char *module,const char *version, const char *logPath = NULL);

	virtual ~TLog();
	//打开日志文件
	FILE *open(int mode = LOG_TO_FILE,int printVersion = 1);
	//关闭日志文件
	void close();

	//设置日志文件名，由应用名和模块名组成
	CTbool SetFileName(const char *appname, const char *module);

	//设置应用模块的版本号
	void SetVersion(const char *szVersion);

	//设置日志的路径
	CTbool SetFilePath(const char *szFilePath);

	//设置日志的输出方式：到nul,FILE（缺省）,SCREEN or both
	void setMode(int mode);

#ifdef __LOG4J__
	//打开log4j的备分命名方式
	void useLog4jBackupStyle();
	static void setMaxBackupFile(CTuint uMaxBackupFile);
#endif

	//设置是否逐行刷新日志标志,0=不逐行刷新，1=逐行刷新（缺省）
	void setFlushFlag(int flag){ m_nFlushFlag = flag;}

	//设置是否打印毫秒,0=不逐行刷新，1=逐行刷新（缺省）
	void setWithMillisecond(bool flag){ m_hasMillilSecond = flag;}

	//设置日志文件最大行数
	void setMaxLines(int nMaxLines);
	
	//设置日志文件备份的时间间隔类型
	void setBackupTimeIntervalType(int backupType);
	//得到日志的输出方式
	int getMode() const { return m_nTraceMode; }

	//普通日志，开头打印时间
	//log4j模式下的日志，精确到毫秒(仅支持LINUX，WIN下真正毫秒级需要时间去算，不建议)，且级别短语自动添加
	void print(CTLogLevel level, const char *format,...);

	//普通日志，开头打印时间和ID
	//log4j模式下的日志，精确到毫秒(仅支持LINUX，WIN下真正毫秒级需要时间去算，不建议)，且级别短语自动添加
	void print(CTLogLevel level, int ID, const char *format,...);

	//普通日志，开头打印时间、ID、qualifier
	//format: MMDD HH:MI:SS <qualifier>[ID]
	//log4j模式下的日志，精确到毫秒(仅支持LINUX，WIN下真正毫秒级需要时间去算，不建议)，且级别短语自动添加
	void print(CTLogLevel level, CTuint qualifier,CTuint ID, const char *format,...);

	//普通日志，开头不打印时间
	//log4j模式下的日志，精确到毫秒(仅支持LINUX，WIN下真正毫秒级需要时间去算，不建议)，且级别短语自动添加
	void printnt(CTLogLevel level, const char *format,...);

#ifdef __LOG4J__
	//普通日志，开头不打印时间和日志级别
	//log4j模式下的日志，精确到毫秒(仅支持LINUX，WIN下真正毫秒级需要时间去算，不建议)
	void printntl(const char *format,...);
#endif

	//二进制日志，开头打印时间
	//log4j模式下的日志，精确到毫秒(仅支持LINUX，WIN下真正毫秒级需要时间去算，不建议)，且级别短语自动添加
	void printb(CTLogLevel level, const char *title, const unsigned char *buf,int len);

	//二进制日志，开头打印时间
	//log4j模式下的日志，精确到毫秒(仅支持LINUX，WIN下真正毫秒级需要时间去算，不建议)，且级别短语自动添加
	void printb(CTLogLevel level, unsigned int ID, const char *title, const unsigned char *buf,int len);

	//二进制日志，开头不打印时间
	//log4j模式下的日志，精确到毫秒(仅支持LINUX，WIN下真正毫秒级需要时间去算，不建议)，且级别短语自动添加
	void printnb(CTLogLevel level, const unsigned char *buf,int len);

	//可以传递变长参数指针进行打印,开头打印时间和ID
	void vprint(CTLogLevel level, int ID, const char *format, va_list argptr);

	//format:<guid>[ID]content...
	void vprint(CTLogLevel level, CTuint guid, CTuint ID, const char *format, va_list argptr);

	//打印错误日志，开头打印时间
	void perror(const char *format,...);

	//刷新日志
	void flush();

	void SetTraceLevelThreshold(int level);	//用于动态调整日志级别
	int GetTraceLevelThreshold(){return m_iTraceLevelThreshold;}

	//打印任意长度的数据 add by oroming in 2006.07.21
	void printBuffer(CTLogLevel level, const char* pTitle,const char* pBuffer,size_t nLen,CTuint id=0);
	void printBuffer(CTLogLevel level, const char* pBuffer,size_t nLen);
	//////////////////////////////////////////////////////////////////////////
	//注册日志事件处理器
	//hr	2007.02.26
	void registerEventHandler(ILogEventHandler* pHandler);

	void WriteUnicodeFlag();

	//普通日志，开头打印时间和ID
	//log4j模式下的日志，精确到毫秒(仅支持LINUX，WIN下真正毫秒级需要时间去算，不建议)，且级别短语自动添加
	void print(CTLogLevel level, const wchar_t *content);
private:
	//防止错误用法：printnt(0, "%d", 1234);
	void printnt(int , const char *format,...);
	void perror(int , const char *format,...);

	void printb(const unsigned char *buf, int len);

	//备份
	void Backup();

	//打印版本信息
	void PrintVersion(void);
	CTbool CanPrint(CTLogLevel level);			//用于控制是否打印日志
	CTbool IsNeedBackup();

	inline void LOCK()  { if(m_nMultiUser)  m_Lock.Lock(); }
	inline void UNLOCK()  { if(m_nMultiUser)  m_Lock.Unlock(); }

	int GetMillisecond();


private:
	FILE *m_FP;							//文件描述符
	CTCriticalSection m_Lock;
#ifdef __LOG4J__
	int m_nUseLog4jStyle; //默认采用磐石的_BAK备份模式,启用的话，就用*.log.n的模式了
	int m_nLastBakupIndex; //范围[1,10] 1.重启的时候更新为目前目录下最大的序号 2.正常的时候就循环递增
	CTuint m_nTotalSize;				//总大小计数
	CTuint m_nMaxSize;					//日志文件的最大大小
	static CTuint m_nMaxBackupFile;
#endif
	char m_acFileName[MAX_PATH];		//日志文件名
	char m_acBakFileName[MAX_PATH];		//备份日志命LOG_Bak
	char m_acErrorFileName[MAX_PATH];	//错误日志文件名
	CTuint m_nTotalCount;				//总行数计数
	CTuint m_nMaxLines;					//日志文件的最大行数
	int m_nTraceMode;					//日志输出位置：nul,文件,屏幕,both
	int m_nFlushFlag;					//是否逐行刷新日志
	int m_nMultiUser;					//是否多用户。多用户使用时需要加锁访问，单用户使用时不需要加锁
	char m_acVersion[MAX_VERSION_LEN];	//版本信息
	int m_iTraceLevelThreshold;
	ILogEventHandler*	m_pEventHandler;//日志事件处理器
	time_t				m_backupFileBeginTime;//备份日志文件的开始时间
	time_t				m_backupFileEndTime;//备份日志文件的结束时间
	CTstring			m_appName;//应用名
	CTstring			m_moduleName;//模块名
	int m_isPrintVersion;  // 日志文件的首行打印版本信息

	bool m_hasMillilSecond;
	int m_bakcupTimeIntervalType;
private:
	TLog(const TLog& rhs);
	TLog& operator=(const TLog& rhs);
}; 
	
class TLogable
{
public:	
	void registerLogger(TLog* pLog) { m_pLog = pLog;}		
protected:
	TLogable():
		m_pLog(NULL)
		{
		}	

protected:		
	TLog* m_pLog;
};
	
#endif //__LOG_H__

