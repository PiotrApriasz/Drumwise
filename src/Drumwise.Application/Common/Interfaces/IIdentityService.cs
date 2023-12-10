using Drumwise.Application.Common.Models;

namespace Drumwise.Application.Common.Interfaces;

public interface IIdentityService
{
    Task<string?> GetUserNameAsync(string userId);

    Task<bool> IsInRoleAsync(string userId, string role);

    Task<bool> AuthorizeAsync(string userId, string policy);

    Task<Result> DeleteUserAsync(string userId);
}