namespace Drumwise.Application.Common.Interfaces;

public interface IMailSender
{
    Task SendMailAsync(string email, string subject, IMailData mailData, string bodyType = "html");
}