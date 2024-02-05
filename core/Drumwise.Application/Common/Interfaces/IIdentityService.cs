using System.Security.Claims;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Common.Models.Identity;

namespace Drumwise.Application.Common.Interfaces;

public interface IIdentityService
{
    Task<string?> GetUserNameAsync(string userId);

    Task<Result> RegisterAdditionalUserData(AdditionalUserDataRequest additionalUserDataRequest,
        ClaimsPrincipal claimsPrincipal);

    Task<Result> DeleteUserAsync(string userId);
}